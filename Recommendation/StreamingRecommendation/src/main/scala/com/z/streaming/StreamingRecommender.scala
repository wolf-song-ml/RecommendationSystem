package com.z.streaming

import com.mongodb.casbah.commons.MongoDBObject
import com.mongodb.casbah.{MongoClient, MongoClientURI}
import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.kafka010.{ConsumerStrategies, KafkaUtils, LocationStrategies}
import org.apache.spark.streaming.{Seconds, StreamingContext}
import redis.clients.jedis.Jedis

// 定义连接助手对象，序列化
object ConnHelper extends Serializable {
  lazy val jedis = new Jedis("192.168.0.241")
  lazy val mongoClient = MongoClient(MongoClientURI("mongodb://192.168.0.241:27017/recommender"))
}

case class MongoConfig(uri: String, db: String)

// 定义一个基准推荐对象
case class Recommendation(mid: Int, score: Double)

// 定义基于预测评分的用户推荐列表
case class UserRecs(uid: Int, recs: Seq[Recommendation])

// 定义基于LFM电影特征向量的电影相似度列表
case class MovieRecs(mid: Int, recs: Seq[Recommendation])

object StreamingRecommender {

  val MAX_USER_RATINGS_NUM = 20
  val MAX_SIM_MOVIES_NUM = 20
  val MONGODB_STREAM_RECS_COLLECTION = "StreamRecs"
  val MONGODB_RATING_COLLECTION = "Rating"
  val MONGODB_MOVIE_RECS_COLLECTION = "MovieRecs"

  def main(args: Array[String]): Unit = {
    val config = Map(
      "mongo.uri" -> "mongodb://192.168.0.241:27017/recommender",
      "mongo.db" -> "recommender",
      "kafka.topic" -> "recommender"
    )

    // 创建一个sparkConf
    val warehouseLocation : String = "hdfs://node1:9000/user/hive/warehouse"
    val sparkConf = new SparkConf()
      .setAppName("推荐系统 - 实时推荐")
      .setMaster("spark://node1:7077,node3:7077")
      .setJars(List("G:\\JavaEE\\Hadoop-Spark\\RecommendationSystem\\Recommendation\\StreamingRecommendation\\target\\" +
        "StreamingRecommendation-jar-with-dependencies.jar"))
      .setIfMissing("spark.driver.host", "192.168.0.28")
      .set("spark.num.executors", "3")
      .set("spark.executor.cores", "2")
      .set("spark.executor.memory", "1800m")
      .set("spark.sql.warehouse.dir", warehouseLocation)

    // 创建一个SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    // 拿到streaming context
    val sc = spark.sparkContext
    val ssc = new StreamingContext(sc, Seconds(2)) // batch duration

    import spark.implicits._
    implicit val mongoConfig = MongoConfig(config("mongo.uri"), config("mongo.db"))

    // 加载电影相似度矩阵数据，把它广播出去
    val simMovieMatrix = spark.read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_MOVIE_RECS_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[MovieRecs]
      .rdd
      .map{ movieRecs => // 为了查询相似度方便，转换成map
        (movieRecs.mid, movieRecs.recs.map( x=> (x.mid, x.score) ).toMap )
      }.collectAsMap()

    val simMovieMatrixBroadCast = sc.broadcast(simMovieMatrix)

    // 定义kafka连接参数:
    val kafkaParam = Map(
      "bootstrap.servers" -> "node1:9092,node2:9092,node3:9092",
      "key.deserializer" -> classOf[StringDeserializer],
      "value.deserializer" -> classOf[StringDeserializer],
      "auto.offset.reset" -> "latest",
      "group.id" -> "recommender"
//      "enable.auto.commit" -> false,
//     "receive.buffer.bytes" -> 65536
    )
    // 通过kafka创建一个DStream
    val kafkaStream = KafkaUtils.createDirectStream[String, String](ssc, LocationStrategies.PreferConsistent,
      ConsumerStrategies.Subscribe[String, String](Array(config("kafka.topic")), kafkaParam)
    )

    // 把原始数据UID|MID|SCORE|TIMESTAMP 转换成评分流
    val ratingStream = kafkaStream.map {
      msg =>
        // """|""":scala中"""三个引号内可以直接敲回车替代\n，stripMargin取空格链接字符默认是|
        val attr = msg.value().split("\\|")
        (attr(0).toInt, attr(1).toInt, attr(2).toDouble, attr(3).toInt)
    }

    ratingStream.print()

    ratingStream.foreachRDD {
      rdd => rdd.foreach {
          case (uid, mid, score, timestamp) => {
            // 从redis里获取用户最近的K次评分:Array[(mid, score)]
            val userRecentlyRatings = getUserRecentlyRating(MAX_USER_RATINGS_NUM, uid, ConnHelper.jedis)

            // 从相似度矩阵中获取备选列表，Array[mid]
            val candidateMovies = getTopSimMovies(MAX_SIM_MOVIES_NUM, mid, uid, simMovieMatrixBroadCast.value)

            // 计算备选元素与用户最近评分物品相似度+加强减弱因子，Array[(mid, score)]
            val streamRecs = computeMovieScores(candidateMovies, userRecentlyRatings, simMovieMatrixBroadCast.value)

            // 数据保存到mongodb
            saveDataToMongoDB(uid, streamRecs)
          }
        }
    }
    // 开始接收和处理数据
    ssc.start()

    println(">>>>>>>>>>>>>>> streaming started!")

    ssc.awaitTermination()

  }

  /**
    * 从redis获取用户最近k次评分
    * @param num 数量
    * @param uid 用户id
    * @param jedis
    * @return
    */
  def getUserRecentlyRating(num: Int, uid: Int, jedis: Jedis): Array[(Int, Double)] = {
    // java list to scala.BufferList
    import scala.collection.JavaConversions._
    // key{uid:UID}, value{MID:SCORE}
    jedis.lrange("uid:" + uid, 0, num - 1)
      .map {
        item =>
          val attr = item.split("\\:")
          (attr(0).trim.toInt, attr(1).trim.toDouble)
      }
      .toArray
  }

  /**
    * 相似从相似度矩阵中获取备选列表：过滤已评分的
    * @param num       相似电影的数量
    * @param mid       当前电影ID
    * @param uid       当前评分用户ID
    * @param simMovies 相似度矩阵
    * @return
    */
  def getTopSimMovies(num: Int, mid: Int, uid: Int, simMovies: scala.collection.Map[Int, scala.collection.immutable.Map[Int, Double]])
                     (implicit mongoConfig: MongoConfig): Array[Int] = {
    val allSimMovies = simMovies(mid).toArray

    val ratingExist = ConnHelper.mongoClient(mongoConfig.db)(MONGODB_RATING_COLLECTION)
      .find(MongoDBObject("uid" -> uid))
      .map {
        item => item.get("mid").toString.toInt
      }.toArray

    allSimMovies.filter(x => !ratingExist.contains(x._1))
      .sortWith(_._2 > _._2)
      .take(num)
      .map(x => x._1)
  }

  /**
    * 计算备选元素与用户最近评分物品相似度+加强减弱因子:核心算法
    * @param candidateMovies
    * @param userRecentlyRatings
    * @param simMovies
    * @return
    */
  def computeMovieScores(candidateMovies: Array[Int], userRecentlyRatings: Array[(Int, Double)],
                         simMovies: scala.collection.Map[Int, scala.collection.immutable.Map[Int, Double]]): Array[(Int, Double)] = {
    val scores = scala.collection.mutable.ArrayBuffer[(Int, Double)]()
    // 增强减弱因子
    val increMap = scala.collection.mutable.HashMap[Int, Int]()
    val decreMap = scala.collection.mutable.HashMap[Int, Int]()

    for (candidateMovie <- candidateMovies; userRecentlyRating <- userRecentlyRatings) {
      val simScore = getMoviesSimScore(candidateMovie, userRecentlyRating._1, simMovies)

      if (simScore > 0.7) {
        scores += ((candidateMovie, simScore * userRecentlyRating._2))
        if (userRecentlyRating._2 > 3) {
          increMap(candidateMovie) = increMap.getOrElse(candidateMovie, 0) + 1
        } else {
          decreMap(candidateMovie) = decreMap.getOrElse(candidateMovie, 0) + 1
        }
      }
    }

    scores.groupBy(_._1).map {
      case (mid, scoreList) =>
        (mid, scoreList.map(_._2).sum / scoreList.length + log(increMap.getOrElse(mid, 1)) - log(decreMap.getOrElse(mid, 1)))
    }.toArray.sortWith(_._2 > _._2)
  }

  /**
    * 物品间相似度：通过查找已计算的相似矩阵（broadCast map结构很方便）
    * @param mid1
    * @param mid2
    * @param simMovies
    * @return
    */
  def getMoviesSimScore(mid1: Int, mid2: Int, simMovies: scala.collection.Map[Int,
    scala.collection.immutable.Map[Int, Double]]): Double = {

    simMovies.get(mid1) match {
      case Some(sims) => sims.get(mid2) match {
        case Some(score) => score
        case None => 0.0
      }
      case None => 0.0
    }
  }

  /**
    * 求对数
    * @param m
    * @return
    */
  def log(m: Int): Double = {
    val N = 10
    math.log(m) / math.log(N)
  }

  def saveDataToMongoDB(uid: Int, streamRecs: Array[(Int, Double)])(implicit mongoConfig: MongoConfig): Unit = {
    // 定义到StreamRecs表的连接
    val streamRecsCollection = ConnHelper.mongoClient(mongoConfig.db)(MONGODB_STREAM_RECS_COLLECTION)

    // 如果表中已有uid对应的数据，则删除
    streamRecsCollection.findAndRemove(MongoDBObject("uid" -> uid))
    // 将streamRecs数据存入表中
    streamRecsCollection.insert(MongoDBObject("uid" -> uid,
      "recs" -> streamRecs.map(x => MongoDBObject("mid" -> x._1, "score" -> x._2))))
  }

}
