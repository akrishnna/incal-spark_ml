package useFunctions

import org.incal.spark_ml.MLResultUtil
import org.incal.spark_ml.models.classification.ClassificationEvalMetric
import org.incal.spark_ml.models.result.ClassificationResultsHolder

object calculateMetrics{
  def calcMeanAccuracy(results: ClassificationResultsHolder) = {
    val metricStatsMap = MLResultUtil.calcMetricStats(results.performanceResults)
    val (trainingAccuracy, Some(testAccuracy), _) = metricStatsMap.get(ClassificationEvalMetric.accuracy).get
    (trainingAccuracy.mean, testAccuracy.mean)
  }
  def calcMeanF1(results: ClassificationResultsHolder) = {
    val metricStatsMap = MLResultUtil.calcMetricStats(results.performanceResults)
    val (trainingAccuracy, Some(testAccuracy), _) = metricStatsMap.get(ClassificationEvalMetric.f1).get
    (trainingAccuracy.mean, testAccuracy.mean)
  }
  def calcMeanWeightedPrecision(results: ClassificationResultsHolder) = {
    val metricStatsMap = MLResultUtil.calcMetricStats(results.performanceResults)
    val (trainingAccuracy, Some(testAccuracy), _) = metricStatsMap.get(ClassificationEvalMetric.weightedPrecision).get
    (trainingAccuracy.mean, testAccuracy.mean)
  }
  def calcMeanWeightedRecall(results: ClassificationResultsHolder) = {
    val metricStatsMap = MLResultUtil.calcMetricStats(results.performanceResults)
    val (trainingAccuracy, Some(testAccuracy), _) = metricStatsMap.get(ClassificationEvalMetric.weightedRecall).get
    (trainingAccuracy.mean, testAccuracy.mean)
  }
  def calcMeanAreaUnderROC(results: ClassificationResultsHolder) = {
    val metricStatsMap = MLResultUtil.calcMetricStats(results.performanceResults)
    val (trainingAccuracy, Some(testAccuracy), _) = metricStatsMap.get(ClassificationEvalMetric.areaUnderROC).get
    (trainingAccuracy.mean, testAccuracy.mean)
  }
  def calcMeanAreaUnderPR(results: ClassificationResultsHolder) = {
    val metricStatsMap = MLResultUtil.calcMetricStats(results.performanceResults)
    val (trainingAccuracy, Some(testAccuracy), _) = metricStatsMap.get(ClassificationEvalMetric.areaUnderPR).get
    (trainingAccuracy.mean, testAccuracy.mean)
  }
}