package useFunctions

import org.incal.core.util.STuple3
import org.incal.spark_ml.models.result.ClassificationResultsHolder

import scala.math.sqrt

object postprocess{

  def create4x4Confusion(classification:Traversable[STuple3[Seq[(Int, Int)]]], normalize:Boolean = false):(Array[Array[Double]],Array[Array[Double]]) ={


    //Create 4x4 zeros array
    var confusion1:Array[Array[Double]] = Array(Array(0.0,0.0,0.0,0.0),Array(0.0,0.0,0.0,0.0),Array(0.0,0.0,0.0,0.0),Array(0.0,0.0,0.0,0.0))
    var confusion2:Array[Array[Double]] = Array(Array(0.0,0.0,0.0,0.0),Array(0.0,0.0,0.0,0.0),Array(0.0,0.0,0.0,0.0),Array(0.0,0.0,0.0,0.0))

    var numReps = classification.toList.length

    //Count each classification case into confusion matrix
    for (i <- 0 to numReps-1){
      var numPairs1 = classification.toList(i)._1.length
      var numPairs2 = classification.toList(i)._2.length

      for (j <- 0 to numPairs1-1){
        var x = classification.toList(i)._1(j)._1
        var y = classification.toList(i)._1(j)._2

        confusion1(x)(y) = confusion1(x)(y) + 1.0
      }

      for (j <- 0 to numPairs2-1){
        var x = classification.toList(i)._2(j)._1
        var y = classification.toList(i)._2(j)._2

        confusion2(x)(y) = confusion2(x)(y) + 1.0
      }

      //Normalize matrix elements if normalize is set to true
      if(normalize) {
        var sum = 0.0
        for (i <- 0 to confusion2.length - 1) {
          for (j <- 0 to confusion2(i).length - 1) {
            sum = sum + confusion2(i)(j).toDouble
          }
        }

        for (i <- 0 to confusion2.length - 1) {
          for (j <- 0 to confusion2(i).length - 1) {
            confusion2(i)(j) = confusion2(i)(j) / sum
          }
        }

        sum = 0.0
        for (i <- 0 to confusion1.length - 1) {
          for (j <- 0 to confusion1(i).length - 1) {
            sum = sum + confusion1(i)(j).toDouble
          }
        }

        for (i <- 0 to confusion1.length - 1) {
          for (j <- 0 to confusion1(i).length - 1) {
            confusion1(i)(j) = confusion1(i)(j) / sum
          }
        }
      }
    }
  return (confusion1, confusion2)
  }

  def printConfusion(confusion:Array[Array[Double]], matrixName:String="Confusion Matrix"): Unit ={
    println(matrixName)
    for(array<-confusion.toList){
      println(array.mkString("\t"))
    }
  }


  def calcSD(sample:List[Double]):Double ={
    //Calculate standard deviation of list of numbers
    var sum = 0.0

    for (element <- sample){
      sum = sum + element
    }


    val mean = sum / sample.length.toDouble
    var sumSquaredError = 0.0

    for (element <- sample){
      sumSquaredError = sumSquaredError + ((element - mean)*(element - mean))
    }


    val SD = sqrt(sumSquaredError / sample.length)

    return SD
  }

  def calcConfidence(results:ClassificationResultsHolder, classifications:Traversable[STuple3[Seq[(Int, Int)]]], metricName:String):(Double, Double) ={

    //Count number of examples in training and test sets
    var testN = 0.0
    var trainN = 0.0

    for (rep <- classifications.toList){
      testN = testN + rep._2.length.toDouble
    }

    for (rep <- classifications.toList){
      trainN = trainN + rep._1.length.toDouble
    }

    //Set correct index for corresponding metric
    var metricIDX = 0
    if (metricName == "weightedRecall"){metricIDX = 0}
    else if (metricName == "weightedPrecision"){metricIDX = 1}
    else if (metricName == "accuracy"){metricIDX = 2}
    else if (metricName == "f1"){metricIDX = 3}
    else {
      println(metricName + " not found.")
      println("Use one of the following metrics:")
      println("weightedRecall")
      println("weightedPrecision")
      println("accuracy")
      println("f1")
      sys.exit()}

    //Create list of metric values of each rep for training and test set
    var trainMetrics:List[Double] = List()
    var testMetrics:List[Double] = List()
    for (rep <- results.performanceResults.toList(metricIDX).trainingTestReplicationResults.toList){
      if (trainMetrics.length == 0 && testMetrics.length == 0){
        trainMetrics = List(rep._1)
        testMetrics = List(rep._2.getOrElse(0.0))
      } else{
        trainMetrics = trainMetrics:::List(rep._1)
        testMetrics = testMetrics:::List(rep._2.getOrElse(0.0))
      }

    }

    //Calculate standard deviation
    val trainSD = calcSD(trainMetrics)
    val testSD = calcSD(testMetrics)

    //Calculate 95% confidence interval
    val trainConfidence = 1.96 * trainSD / sqrt(trainMetrics.length.toDouble)
    val testConfidence = 1.96 * testSD / sqrt(testMetrics.length.toDouble)

    return (trainConfidence, testConfidence)
  }

}