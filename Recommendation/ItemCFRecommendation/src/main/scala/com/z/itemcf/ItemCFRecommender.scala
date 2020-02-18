package com.z.itemcf

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

case class MongoConfig( uri: String, db: String )

case class MovieRating( uid: Int, mid: Int, score: Double, timestamp: Int )

case class Recommendation( mid: Int, score: Double )

case class MoviesRecs( mid: Int, recs: Seq[Recommendation] )

object ItemCFRecommender {
  // 定义常量和表名
  val MONGODB_RATING_COLLECTION = "Rating"
  val ITEM_CF_MOVIE_RECS = "ItemCFMoviesRecs"
  val MAX_RECOMMENDATION = 10

  def main(args: Array[String]): Unit = {
    val config = Map(
      "mongo.uri" -> "mongodb://192.168.0.241:27017/recommender",
      "mongo.db" -> "recommender"
    )
    // 创建一个sparkConf
    val warehouseLocation : String = "hdfs://node1:9000/user/hive/warehouse"
    val sparkConf = new SparkConf()
      .setAppName("推荐系统 - itemCF")
      .setMaster("spark://node1:7077,node3:7077")
      .setJars(List("G:\\JavaEE\\Hadoop-Spark\\RecommendationSystem\\Recommendation\\ItemCFRecommendation\\target\\" +
        "ItemCFRecommendation-jar-with-dependencies.jar"))
      .setIfMissing("spark.driver.host", "192.168.0.28")
      .set("spark.num.executors", "3")
      .set("spark.executor.cores", "2")
      .set("spark.executor.memory", "1800m")
      .set("spark.sql.warehouse.dir", warehouseLocation)

    // 创建spark session
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    import spark.implicits._
    implicit val mongoConfig = MongoConfig( config("mongo.uri"), config("mongo.db") )

    // 加载数据，转换成DF进行处理
    val ratingDF = spark.read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_RATING_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[MovieRating]
      .map(
        x => ( x.uid, x.mid, x.score )
      )
      .toDF("uid", "mid", "score")
      .cache()

    val productRatingCountDF = ratingDF.groupBy("mid").count() //默认clos as:count
    val ratingWithCountDF = ratingDF.join(productRatingCountDF, "mid")

    // 核心算法：
    val joinedDF = ratingWithCountDF.join(ratingWithCountDF, "uid") // .where($"mid" != $"mid")
      .toDF("uid","mid1","score1","count1","mid2","score2","count2")
      .select("uid","mid1","count1","mid2","count2").where($"mid1" =!= $"mid2")

    joinedDF.createOrReplaceTempView("joined")
    // scala """ | stripMargin妙用.注意string.spilit("""|""")
    val cooccurrenceDF = spark.sql(
      """
        |select mid1
        |, mid2
        |, count(uid) as cocount
        |, first(count1) as count1
        |, first(count2) as count2
        |from joined
        |group by mid1, mid2
      """.stripMargin
    ).cache()

    // ( mid1, (mid2, score) )
    val simDF = cooccurrenceDF.map{
      row =>
        val coocSim = cooccurrenceSim( row.getAs[Long]("cocount"), row.getAs[Long]("count1"),
          row.getAs[Long]("count2") )
        ( row.getInt(0), ( row.getInt(1), coocSim ) )
    }
      .rdd
      .groupByKey()
      .map{
        case (mid, recs) =>
          MoviesRecs( mid, recs.toList.sortWith(_._2>_._2).take(MAX_RECOMMENDATION)
            .map(x=>Recommendation(x._1,x._2)) )
      }
      .toDF()

    simDF.write
      .option("uri", mongoConfig.uri)
      .option("collection", ITEM_CF_MOVIE_RECS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    spark.stop()
  }

  // 同现相似度计算公式
  def cooccurrenceSim(coCount: Long, count1: Long, count2: Long): Double ={
    coCount / math.sqrt( count1 * count2 )
  }

}
