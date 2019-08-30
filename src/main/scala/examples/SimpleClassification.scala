package examples

import useFunctions.calculateMetrics._
import useFunctions.preprocess._
import useFunctions.postprocess._

import org.apache.spark.sql.SparkSession
import org.incal.spark_ml.models.TreeCore
import org.incal.spark_ml.models.classification._
import org.incal.spark_ml.{MLResultUtil, SparkMLApp, SparkMLService, SparkMLServiceSetting}
import org.incal.spark_ml.SparkUtil._
import org.incal.spark_ml.models.result.ClassificationResultsHolder
import org.incal.spark_ml.models.setting.ClassificationLearningSetting

import scala.concurrent.ExecutionContext.Implicits.global

object SimpleClassification extends SparkMLApp((session: SparkSession, mlService: SparkMLService) => {

  session.sparkContext.setLogLevel("ERROR")

  object Column extends Enumeration {

    val Exercise = Value

    def createValues(colNames:Array[String], session:SparkSession)={
      val nClass = colNames.size
      for(i <- 1 to nClass){
        if (colNames(i-1) != "Exercise") {
          Value(colNames(i - 1).toString)
        }
      }
    }
  }

  //Dataset paths
  val irisPath = "/home/aditya/Desktop/summer_2019/incal-spark_ml/data_sets/iris/iris.csv"
  val egaitPath =  "/home/aditya/Desktop/summer_2019/incal-spark_ml-master/data_sets/alias/egait.features_filtered.csv"


  //Load dataset in dataframe
  val df0_0 = loadCSV(egaitPath, session)

  //Run preprocessing steps
  val df0_1 = df0_0.drop("SessionId")
  val df1_0 = dropNullCols(df0_1, 1.0, session)
  val df2_0 = removeSqBrackets(df1_0, session)
  val df3_0 = handleArrays(df2_0, session)
  val df4_0 = impute(df3_0, session)
  val df = df4_0

  Column.createValues(df.columns, session)

  val columnNames = Column.values.toSeq.sortBy(_.id).map(_.toString)
  val outputColumnName = Column.Exercise.toString
  val featureColumnNames = columnNames.filter(_ != outputColumnName)


  // read a csv and create a data frame with given column names
  //val url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
  //val df = remoteCsvToDataFrame(url, false)(session).toDF(columnNames :_*)

  // index the clazz column since it's of the string type
  val df2 = indexStringCols(Seq((outputColumnName, Nil)))(df)

  // turn the data frame into ML-ready one with features and a label
  val finalDf = prepFeaturesDataFrame(featureColumnNames.toSet, Some(outputColumnName))(df2)

  // logistic regression spec
  val logisticRegressionSpec = LogisticRegression(
    family = Some(LogisticModelFamily.Multinomial),
    maxIteration = Left(Some(200)),
    regularization = Left(Some(0.001)),
    elasticMixingRatio = Left(Some(0.5))
  )

  // multi-layer perceptron spec
  val multiLayerPerceptronSpec = MultiLayerPerceptron(
    hiddenLayers = Seq(5, 5, 5)
  )

  // random forest spec
  val randomForestSpec = RandomForest(
    core = TreeCore(maxDepth = Left(Some(5)))
  )


  // learning setting of cross-validation
  val learningSetting = ClassificationLearningSetting(
    repetitions = Some(25),
    crossValidationFolds = Some(5),
    crossValidationEvalMetric = Some(ClassificationEvalMetric.accuracy),
    collectOutputs = true
  )

  for {
    // run the logistic regression and get results
    logisticRegressionResults <- mlService.classify(finalDf, logisticRegressionSpec, learningSetting)

    // run the multi-layer perceptron and get results
    //multiLayerPerceptronResults <- mlService.classify(finalDf, multiLayerPerceptronSpec, learningSetting)

    // run the random forest and get results
    //randomForestResults <- mlService.classify(finalDf, randomForestSpec, learningSetting)
  } yield {

    //Change classifierName and results depending on the trained model
    val classifierName = "Logistic regression"
    val results = logisticRegressionResults



    val (lrTrainingAccuracy, lrTestAccuracy) = calcMeanAccuracy(results)
    val (lrTrainingF1, lrTestF1) = calcMeanF1(results)
    val (lrTrainingWeightedPrecision, lrTestWeightedPrecision) = calcMeanWeightedPrecision(results)
    val (lrTrainingWeightedRecall, lrTestWeightedRecall) = calcMeanWeightedRecall(results)

    val (trainingAccuracyConfidence, testAccuracyConfidence) = calcConfidence(results, results.expectedActualOutputs, "accuracy")
    val (trainingF1Confidence, testF1Confidence) = calcConfidence(results, results.expectedActualOutputs, "f1")
    val (trainingWPrecisionConfidence, testWPrecisionConfidence) = calcConfidence(results, results.expectedActualOutputs, "weightedPrecision")
    val (trainingWRecallConfidence, testWRecallConfidence) = calcConfidence(results, results.expectedActualOutputs, "weightedRecall")



    println(s"$classifierName    (accuracy): $lrTrainingAccuracy / $lrTestAccuracy")
    println(s"$classifierName    (confidence): $trainingAccuracyConfidence / $testAccuracyConfidence")

    println(s"$classifierName    (F1): $lrTrainingF1 / $lrTestF1")
    println(s"$classifierName    (confidence): $trainingF1Confidence / $testF1Confidence")

    println(s"$classifierName    (WeightedPrecision): $lrTrainingWeightedPrecision / $lrTestWeightedPrecision")
    println(s"$classifierName    (confidence): $trainingWPrecisionConfidence / $testWPrecisionConfidence")

    println(s"$classifierName    (WeightedRecall): $lrTrainingWeightedRecall / $lrTestWeightedRecall")
    println(s"$classifierName    (confidence): $trainingWRecallConfidence / $testWRecallConfidence")

    val (fullConfusion, testConfusion) = create4x4Confusion(results.expectedActualOutputs)
    printConfusion(fullConfusion, "Full Set Confusion Matrix")
    printConfusion(testConfusion, "Test Set Confusion Matrix")

  }
}) {
  override protected def mlServiceSetting = SparkMLServiceSetting(
    debugMode = false,
    repetitionParallelism = Option(1)
  )
}