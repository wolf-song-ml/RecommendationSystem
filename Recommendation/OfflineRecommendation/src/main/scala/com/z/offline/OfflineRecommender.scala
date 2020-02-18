package com.z.offline

/**
  * 离线推荐算法：
  * 根据用户推荐电影列表
  * 电影相似度矩阵列表
  */

import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.jblas.DoubleMatrix

// Rate表：与ALS算法中的Rating取别开
case class MovieRating(uid: Int, mid: Int, score: Double, timestamp: Int)

// 定义一个基准推荐对象
case class Recommendation(mid: Int, score: Double)

// 定义基于预测评分的用户推荐列表
case class UserRecs(uid: Int, recs: Seq[Recommendation])

// 定义基于LFM电影特征向量的电影相似度列表
case class MovieRecs(mid: Int, recs: Seq[Recommendation])

case class MongoConfig(uri: String, db: String)

object OfflineRecommender {

  // 定义表名和常量
  val MONGODB_RATING_COLLECTION = "Rating"
  val USER_RECS = "UserRecs"
  val MOVIE_RECS = "MovieRecs"
  val USER_MAX_RECOMMENDATION = 20

  def main(args: Array[String]): Unit = {
    val config = Map(
      "mongo.uri" -> "mongodb://192.168.0.241:27017/recommender",
      "mongo.db" -> "recommender"
    )

    // 创建一个sparkConf
    val warehouseLocation = "hdfs://node1:9000/user/hive/warehouse";
    val sparkConf = new SparkConf()
      .setAppName("推荐系统 - 离线推荐")
      .setMaster("spark://node1:7077,node3:7077")
      .setJars(List("G:\\JavaEE\\Hadoop-Spark\\RecommendationSystem\\Recommendation\\OfflineRecommendation\\target\\" +
        "OfflineRecommendation-jar-with-dependencies.jar"))
      .setIfMissing("spark.driver.host", "192.168.0.28")
      .set("spark.sql.warehouse.dir", warehouseLocation)
      .set("spark.num.executors", "3")
      .set("spark.executor.cores", "1")
      .set("spark.executor.memory", "1800m")

    // 创建一个SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    import spark.implicits._
    implicit val mongoConfig = MongoConfig(config("mongo.uri"), config("mongo.db"))

    // 加载评分数据
    val ratingRDD = spark.read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_RATING_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[MovieRating]
      .rdd
      .map(rating => (rating.uid, rating.mid, rating.score))
      .cache() // 多次计算缓存到内存中

    // 训练隐语义模型：Rating(user:Int, product:Int, rating:Double)
    val trainData = ratingRDD.map(x => Rating(x._1, x._2, x._3))
    // 多个变量赋值
    val (rank, iterations, lambda) = (200, 5, 0.1)
    val model = ALS.train(trainData, rank, iterations, lambda)

    // 从rating数据中提取所有的uid和mid，并去重
    val userRDD = ratingRDD.map(_._1).distinct()
    val movieRDD = ratingRDD.map(_._2).distinct()
    val userMovies = userRDD.cartesian(movieRDD)

    // 调用model的predict方法预测评分
    val preRatings = model.predict(userMovies)

    val userRecs = preRatings
      .filter(_.rating > 0)
      .map(rating => (rating.user, (rating.product, rating.rating))) // Rating->(uid, (mid, score))
      .groupByKey()
      .map {
        case (uid, recs) => UserRecs(uid, recs.toList.sortWith(_._2 > _._2).take(USER_MAX_RECOMMENDATION).map(x => Recommendation(x._1, x._2)))
      }
      .toDF()

    storeDFInMongoDB(userRecs, USER_RECS)

    /**
      * vector:
      * local vector是一种索引是0开始的整数、内容为double类型，存储在单机上的向量。MLlib支持两种矩阵，dense密集型和sparse稀疏型。
      * 一个dense类型的向量背后其实就是一个数组，而sparse向量背后则是两个并行数组——索引数组和值数组。比如向量(1.0, 0.0, 3.0)
      * 既可以用密集型向量表示为[1.0, 0.0, 3.0]，也可以用稀疏型向量表示为(3, [0,2],[1.0,3.0])，其中3是数组的大小。
      *
      * dense vector与sparse vector:
      * new DenseVector(this.toArray)
      * 创建dense vector
      * val dv: Vector = Vectors.dense(1.0, 0.0, 3.0)
      * 创建sparse vector
      * val sv1: Vector = Vectors.sparse(3, Array(0,2), Array(1.0,3.0))
      * val sv2: Vector = Vectors.sparse(3, Seq((0, 1.0), (2,3.0)))
      *
      * vecor norm范数和sqdist距离:
      * val norm1Vec = Vectors.dense(1.0,-1.0,2.0)
      * // 第一范数，就是绝对值相加
      * println(Vectors.norm(norm1Vec,1)) // 4.0
      * // 第二番薯，就是平方和开根号
      * println(Vectors.norm(norm1Vec,2)) // 2.449489742783178
      * // 无限范数
      * println(Vectors.norm(norm1Vec,1000)) //2.0
      *
      * val sq1 = Vectors.dense(1.0, 2.0, 3.0)
      * val sq2 = Vectors.dense(2.0, 4.0, 6.0)
      * println(Vectors.sqdist(sq1, sq2)) // (2-1)^2 + (4-2)^2 + (6-3)^2 = 14
      *
      * labeled point:
      * 这种labeled point其实内部也是一个vector，可能是dense也可能是sparse，不过多了一个标签列。在ML里面，labeled point
      * 通常用于有监督算法。这个label是double类型的，这样既可以用于回归算法，也可以用于分类。在二分类中，Label不是0就是1；
      * 在多分类中label可能从0开始，1，2，3，4....
      * val pos = LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0))
      * val neg = LabeledPoint(0.0, Vectors.sparse(3, Array(0, 2), Array(1.0, 3.0)))
      * label index1:value1 index2:value2 ...
      * val examples: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
      **/
    val movieFeatures = model.productFeatures.map {

      case (mid, features) => (mid, new DoubleMatrix(features))
    }

    val movieRecs = movieFeatures.cartesian(movieFeatures)
      .filter {
        case (a, b) => a._1 != b._1
      }
      .map{
        case (a, b) => val simScore = this.consinSim(a._2, b._2);(a._1, (b._1, simScore))
      }
      .filter(_._2._2 > 0.6) // 过滤出相似度大于0.6的
      .groupByKey()
      .map {case (mid, items) => MovieRecs(mid, items.toList.sortWith(_._2 > _._2).map(x => Recommendation(x._1, x._2)))}
      .toDF()

    storeDFInMongoDB(movieRecs, MOVIE_RECS)

    spark.stop()
  }

  /**
    * 求向量余弦相似度：矩阵内积/第二范数乘积
    * 皮尔逊相关系数：先对向量每一分量减去分量均值，再求余弦相似度(叫取中心化)
    * @param movie1
    * @param movie2
    * @return
    */
  def consinSim(movie1: DoubleMatrix, movie2: DoubleMatrix): Double = {
    movie1.dot(movie2) / (movie1.norm2() * movie2.norm2())
  }

  def storeDFInMongoDB(df: DataFrame, collection_name: String)(implicit mongoConfig: MongoConfig): Unit ={
    df.write
      .option("uri", mongoConfig.uri)
      .option("collection", collection_name)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()
  }

}
