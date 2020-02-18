package com.z.offline

import breeze.numerics.sqrt
import com.z.offline.OfflineRecommender.MONGODB_RATING_COLLECTION
import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
//import org.apache.spark.ml.recommendation.{ALS, MatrixFactorizationModel, Rating}

/**
  * 隐语模型超参数调整优化
  */
object ALSTrainer {
  def main(args: Array[String]): Unit = {
    val config = Map(
      "mongo.uri" -> "mongodb://192.168.0.241:27017/recommender",
      "mongo.db" -> "recommender"
    )
    // 创建一个sparkConf
    val warehouseLocation = "hdfs://node1:9000/user/hive/warehouse";
    val sparkConf = new SparkConf()
      .setAppName("推荐系统 - LFM调参cache")
      .setMaster("spark://node1:7077,node3:7077")
      .setJars(List("G:\\JavaEE\\Hadoop-Spark\\MovieRecommendSystem\\recommender\\OfflineRecommender\\target\\" +
        "OfflineRecommender-jar-with-dependencies.jar"))
      .setIfMissing("spark.driver.host", "192.168.0.28")
      .set("spark.sql.warehouse.dir", warehouseLocation)
      .set("spark.num.executors", "3")
      .set("spark.executor.cores", "1")
      .set("spark.executor.memory", "1024m")

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
      .map(rating => Rating(rating.uid, rating.mid, rating.score))
//      .cache()

    // 随机切分数据集，生成训练集和测试集
    val splits = ratingRDD.randomSplit(Array(0.8, 0.2))
    val trainingRDD = splits(0)
    val testRDD = splits(1)

    // 模型参数选择，输出最优参数
    adjustALSParam(trainingRDD, testRDD)

    spark.close()
  }

  /**
    * LFM迭代调参
    * @param trainData
    * @param testData
    */
  def adjustALSParam(trainData: RDD[Rating], testData: RDD[Rating]): Unit = {
    val result = for (rank <- Array(50, 100, 200, 300); lambda <- Array(0.01, 0.1, 1))
      yield {
        val model = ALS.train(trainData, rank, 5, lambda)
        val rmse = getRMSE(model, testData)
        (rank, lambda, rmse)
      }

    // 控制台打印输出最优参数
    println(result.minBy(_._3))
  }

  /**
    * 均方误差的根
    * @param model
    * @param data
    * @return
    */
  def getRMSE(model: MatrixFactorizationModel, data: RDD[Rating]): Double = {
    // 计算预测评分
    val userProducts = data.map(item => (item.user, item.product))
    val predictRating = model.predict(userProducts)

    // 以uid，mid作为外键，inner join实际观测值和预测值
    val actual = data.map(item => ((item.user, item.product), item.rating))
    val predict = predictRating.map(item => ((item.user, item.product), item.rating))

    // 内连接得到(uid, mid),(actual, predict)
    sqrt(
      actual.join(predict).map {
        case ((uid, mid), (actual, pre)) => val err = actual - pre; err * err
      }.mean()
    )
  }

}
