package com.z.content

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.SparkSession
import org.jblas.DoubleMatrix

// 需要的数据源是电影内容信息
case class Movie(mid: Int, name: String, descri: String, timelong: String, issue: String, shoot: String, language: String,
                 genres: String, actors: String, directors: String)

case class MongoConfig(uri:String, db:String)

// 定义一个基准推荐对象
case class Recommendation( mid: Int, score: Double )

// 定义电影内容信息提取出的特征向量的电影相似度列表
case class MovieRecs( mid: Int, recs: Seq[Recommendation] )

object ContentRecommender {

  // 定义表名和常量
  val MONGODB_MOVIE_COLLECTION = "Movie"

  val CONTENT_MOVIE_RECS = "ContentMovieRecs"

  def main(args: Array[String]): Unit = {
    val config = Map(
      "mongo.uri" -> "mongodb://192.168.0.241:27017/recommender",
      "mongo.db" -> "recommender"
    )

    // 创建一个sparkConf
    val warehouseLocation : String = "hdfs://node1:9000/user/hive/warehouse"
    val sparkConf = new SparkConf()
      .setAppName("推荐系统 - 内容推荐")
      .setMaster("spark://node1:7077,node3:7077")
      .setJars(List("G:\\JavaEE\\Hadoop-Spark\\RecommendationSystem\\Recommendation\\ContentRecommendation\\target\\" +
        "ContentRecommendation-jar-with-dependencies.jar"))
      .setIfMissing("spark.driver.host", "192.168.0.28")
      .set("spark.num.executors", "3")
      .set("spark.executor.cores", "2")
      .set("spark.executor.memory", "1800m")
      .set("spark.sql.warehouse.dir", warehouseLocation)

    // 创建一个SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    import spark.implicits._
    implicit val mongoConfig = MongoConfig(config("mongo.uri"), config("mongo.db"))

    // 加载数据，并作预处理
    val movieTagsDF = spark.read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_MOVIE_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[Movie]
      .map(
        // 提取mid，name，genres三项作为原始内容特征，分词器默认按照空格做分词
        x => ( x.mid, x.name, x.genres.map(c=> if(c=='|') ' ' else c) )
      )
      .toDF("mid", "name", "genres")
      .cache()

    // 核心部分： 用TF-IDF从内容信息中提取电影特征向量
    // 创建一个分词器，默认按空格分词
    val tokenizer = new Tokenizer().setInputCol("genres").setOutputCol("words")

    // 用分词器对原始数据做转换，生成新的一列words
    val wordsData = tokenizer.transform(movieTagsDF)

    // 引入HashingTF工具，可以把一个词语序列转化成对应的词频
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(50)
    val featurizedData = hashingTF.transform(wordsData)

    // 引入IDF工具，可以得到idf模型
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    // 训练idf模型，得到每个词的逆文档频率
    val idfModel = idf.fit(featurizedData)
    // 用模型对原数据进行处理，得到文档中每个词的tf-idf，作为新的特征向量
    val rescaledData = idfModel.transform(featurizedData)

    val movieFeatures = rescaledData.map {
      row => (row.getAs[Int]("mid"), row.getAs[SparseVector]("features").toArray)
      }.rdd
      .map {
        x => (x._1, new DoubleMatrix(x._2))
      }

    // 对所有电影两两计算它们的相似度，先做笛卡尔积
    val movieRecs = movieFeatures.cartesian(movieFeatures)
      .filter{
        // 把自己跟自己的配对过滤掉
        case (a, b) => a._1 != b._1
      }
      .map{
        case (a, b) => {
          val simScore = this.consinSim(a._2, b._2)
          ( a._1, ( b._1, simScore ) )
        }
      }
      .filter(_._2._2 > 0.6)    // 过滤出相似度大于0.6的
      .groupByKey()
      .map{
        case (mid, items) => MovieRecs( mid, items.toList.sortWith(_._2 > _._2).map(x => Recommendation(x._1, x._2)) )
      }
      .toDF()

    movieRecs.write
      .option("uri", mongoConfig.uri)
      .option("collection", CONTENT_MOVIE_RECS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    spark.stop()
  }

  // 求向量余弦相似度
  def consinSim(movie1: DoubleMatrix, movie2: DoubleMatrix):Double ={
    movie1.dot(movie2) / ( movie1.norm2() * movie2.norm2() )
  }
}
