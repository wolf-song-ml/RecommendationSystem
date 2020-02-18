package com.z.statistics
/**
  * 离线统计。统计的功能：
  * 电影的评分次数统计：mid，count
  * 按月维度评分排行榜：电影每月评分次数并做时间倒序、评分次数倒序
  * 统计电影的平均评分：mid，avg
  * 各类别电影评分Top10统计
  */
import java.text.SimpleDateFormat
import java.util.Date
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
/**
  * Movie 数据集
  * 260                                         电影ID，mid
  * Star Wars: Episode IV - A New Hope (1977)   电影名称，name
  * Princess Leia is captured and held hostage  详情描述，descri
  * 121 minutes                                 时长，timelong
  * September 21, 2004                          发行时间，issue
  * 1977                                        拍摄时间，shoot
  * English                                     语言，language
  * Action|Adventure|Sci-Fi                     类型，genres
  * Mark Hamill|Harrison Ford|Carrie Fisher     演员表，actors
  * George Lucas                                导演，directors
  */
case class Movie(mid: Int, name: String, descri: String, timelong: String, issue: String, shoot: String, language: String,
                 genres: String, actors: String, directors: String)

/**
  * Rate 电影评分数据集
  * @param uid 用户id
  * @param mid 电影id
  * @param score 评分
  * @param timestamp 评分时间戳
  */
case class Rating(uid: Int, mid: Int, score: Double, timestamp: Int )

// 定义一个基准推荐对象
case class Recommendation( mid: Int, score: Double )

// 定义电影类别top10推荐对象
case class GenresRecommendation(genres: String, recs: Seq[Recommendation])

case class MongoConfig(uri:String, db:String)

object StatisticsRecommender {

  // 定义表名
  val MONGODB_MOVIE_COLLECTION = "Movie"
  val MONGODB_RATING_COLLECTION = "Rating"

  //统计的表的名称
  val RATE_MORE_MOVIES = "RateMoreMovies"
  val RATE_MORE_RECENTLY_MOVIES = "RateMoreRecentlyMovies"
  val AVERAGE_MOVIES = "AverageMovies"
  val GENRES_TOP_MOVIES = "GenresTopMovies"

  def main(args: Array[String]): Unit = {
    val config = Map(
      "mongo.uri" -> "mongodb://192.168.0.241:27017/recommender",
      "mongo.db" -> "recommender"
    )

    // 创建一个sparkConf
    val warehouseLocation = "hdfs://node1:9000/user/hive/warehouse";
    val sparkConf = new SparkConf()
      .setAppName("推荐系统 - 离线统计")
      .setMaster("spark://node1:7077,node3:7077")
      .setJars(List("G:\\JavaEE\\Hadoop-Spark\\RecommendationSystem\\Recommendation\\StatisticsRecommendation\\target\\" +
        "StatisticsRecommendation-jar-with-dependencies.jar"))
      .setIfMissing("spark.driver.host", "192.168.0.28")
      .set("spark.sql.warehouse.dir", warehouseLocation)
      .set("spark.num.executors", "3")
      .set("spark.executor.cores", "2")
      .set("spark.executor.memory", "1024m")

    // 创建一个SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    import spark.implicits._
    implicit val mongoConfig = MongoConfig(config("mongo.uri"), config("mongo.db"))

    // 从mongodb加载数据
    val ratingDF = spark.read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_RATING_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[Rating] // 装换成dataset强类型
      .toDF()

    val movieDF = spark.read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_MOVIE_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[Movie] // 装换成dataset强类型
      .toDF()

    // ratings评分表copy到内存中
    ratingDF.createOrReplaceTempView("ratings")

    // 电影的评分次数统计：mid，count
    val rateMoreMoviesDF = spark.sql("select mid, count(mid) as count from ratings group by mid")
    storeDFInMongoDB(rateMoreMoviesDF, RATE_MORE_MOVIES )

    // 按月维度评分排行榜：电影每月评分次数并做时间倒序、评分次数倒序
    val simpleDateFormat = new SimpleDateFormat("yyyyMM")
    spark.udf.register("changeDate", (x: Int) => simpleDateFormat.format(new Date(x * 1000L)).toInt)
    val ratingOfYearMonth = spark.sql("select mid, score, changeDate(timestamp) as yearmonth from ratings")
    ratingOfYearMonth.createOrReplaceTempView("ratingOfMonth")
    val rateMoreRecentlyMoviesDF = spark.sql("select mid, count(mid) as count, yearmonth from ratingOfMonth" +
      " group by yearmonth, mid order by yearmonth desc, count desc")
    storeDFInMongoDB(rateMoreRecentlyMoviesDF, RATE_MORE_RECENTLY_MOVIES)

    // 统计电影的平均评分：mid，avg
    val averageMoviesDF = spark.sql("select mid, avg(score) as avg from ratings group by mid")
    storeDFInMongoDB(averageMoviesDF, AVERAGE_MOVIES)

    // 类别下热门电影榜:对比hive与rdd实现
    /**
      * select * from (
      *   select mid, score, genres_name from movie_with_score lateral view explode(genres) table_tmp as genres_name
      *   ) t row_number over(partition by genres_name order by score desc) rank
      *   where rank <=10
      */
    val genres = List("Action","Adventure","Animation","Comedy","Crime","Documentary","Drama","Family","Fantasy","Foreign",
      "History","Horror","Music","Mystery" ,"Romance","Science","Tv","Thriller","War","Western")
    val movieWithScore = movieDF.join(averageMoviesDF, "mid")
    // movieWithScore.agg($"avg".as("score"))
    val genresRDD = spark.sparkContext.makeRDD(genres)

    // DataFrame->RDD, 内容是Row
    val genresTopMoviesDF = genresRDD.cartesian(movieWithScore.rdd)
      .filter{case (genre, movieRow) =>
        movieRow.getAs[String]("genres").toLowerCase.contains(genre.toLowerCase)
      }
      .map{
        case (genre, movieRow) => (genre, (movieRow.getAs[Int]("mid"), movieRow.getAs[Double]("avg")))
      }
      .groupByKey()
      .map{case (genre, items) =>
        GenresRecommendation(genre, items.toList.sortWith(_._2>_._2).take(10).map(item=> Recommendation(item._1, item._2)))
      }
      .toDF()

    storeDFInMongoDB(genresTopMoviesDF, GENRES_TOP_MOVIES)

    spark.stop()
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
