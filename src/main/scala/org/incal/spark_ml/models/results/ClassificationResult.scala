package org.incal.spark_ml.models.results

import java.{util => ju}

import reactivemongo.bson.BSONObjectID
import org.incal.spark_ml.models.VectorScalerType
import org.incal.spark_ml.models.LearningSetting
import org.incal.spark_ml.models.classification.ClassificationEvalMetric

case class ClassificationResult(
  _id: Option[BSONObjectID],
  setting: ClassificationSetting,
  trainingStats: ClassificationMetricStats,
  testStats: ClassificationMetricStats,
  replicationStats: Option[ClassificationMetricStats] = None,
  trainingBinCurves: Seq[BinaryClassificationCurves] = Nil,
  testBinCurves: Seq[BinaryClassificationCurves] = Nil,
  replicationBinCurves: Seq[BinaryClassificationCurves] = Nil,
  timeCreated: ju.Date = new ju.Date()
)

case class ClassificationMetricStats(
  f1: MetricStatsValues,
  weightedPrecision: MetricStatsValues,
  weightedRecall: MetricStatsValues,
  accuracy: MetricStatsValues,
  areaUnderROC: Option[MetricStatsValues],
  areaUnderPR: Option[MetricStatsValues]
)

case class BinaryClassificationCurves(
  // ROC - FPR vs TPR (false positive rate vs true positive rate)
  roc: Seq[(Double, Double)],
  // PR - recall vs precision
  precisionRecall: Seq[(Double, Double)],
  // threshold vs F-Measure: curve with beta = 1.0.
  fMeasureThreshold: Seq[(Double, Double)],
  // threshold vs precision
  precisionThreshold: Seq[(Double, Double)],
  // threshold vs recall
  recallThreshold: Seq[(Double, Double)]
)

case class ClassificationSetting(
  mlModelId: BSONObjectID,
  outputFieldName: String,
  inputFieldNames: Seq[String],
  filterId: Option[BSONObjectID] = None,
  featuresNormalizationType: Option[VectorScalerType.Value] = None,
  featuresSelectionNum: Option[Int] = None,
  pcaDims: Option[Int] = None,
  trainingTestSplitRatio: Option[Double] = None,
  replicationFilterId: Option[BSONObjectID] = None,
  samplingRatios: Seq[(String, Double)],
  repetitions: Option[Int] = None,
  crossValidationFolds: Option[Int] = None,
  crossValidationEvalMetric: Option[ClassificationEvalMetric.Value] = None,
  binCurvesNumBins: Option[Int] = None
) {
  def fieldNamesToLoads =
    if (inputFieldNames.nonEmpty) (inputFieldNames ++ Seq(outputFieldName)).toSet.toSeq else Nil

  def learningSetting =
    LearningSetting[ClassificationEvalMetric.Value](featuresNormalizationType, pcaDims, trainingTestSplitRatio, samplingRatios, repetitions, crossValidationFolds, crossValidationEvalMetric)
}