package useFunctions

import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.incal.core.util.STuple3
import org.apache.spark.sql.functions._

object preprocess{

  def loadCSV(path:String, spark:SparkSession):DataFrame ={

    //Loads file with pre-defined settings
    val df = spark.read
      .format("csv")
      .option("header", "true")
      .option("delimiter", "\t")
      .option("inferSchema", "true")
      .option("nullValue", "[null]")
      .option("path", path)
      .load()

    return df
  }

  def countNulls(df:DataFrame, spark:SparkSession):List[(String, Int)] ={

    //Count number of nulls in each column
    val colNames = df.columns
    var nullCount = List[(String, Int)]()

    for (column <- colNames){
      var counter = df.filter(col(column).isNull).count()
      nullCount = nullCount ::: List((column, counter.toInt))
    }

    return nullCount
  }

  def dropNullCols(df:DataFrame, thr:Double, spark:SparkSession):DataFrame ={

    //Calculate number of nulls allowed in column
    var nullCount = countNulls(df, spark)
    var scaledTHR = thr*df.count().toDouble
    var colsToRemove = Seq[String]()

    //Collect column names with number of nulls exceeding threshold
    nullCount.foreach{
      case (column, numNulls) => {
        if (numNulls > scaledTHR.toInt){
          colsToRemove = colsToRemove :+ column.toString
        }
      }
    }

    //Drop columns
    val dfNew = df.drop(colsToRemove:_*)

    return dfNew
  }

  def mergeDataFrames(df1:DataFrame, df2:DataFrame, spark:SparkSession):DataFrame = {

    //Join two dataframes by given order (no common features)
    val df1_ = df1.withColumn("id", monotonically_increasing_id())
    var df2_ = df2.withColumn("id", monotonically_increasing_id())
    val df3  = df1_.join(df2_, df2_("id")===df1_("id"), "left_outer")
    val df4 = df3.drop("id")

    return df4
  }

  def lengthUniform(df:DataFrame, spark:SparkSession):Boolean ={

    //Check for uniform number of values in a column per row
    val numRows = df.count().toInt
    val stringValues = df.take(numRows).mkString(",,").split(",,")
    val length = stringValues(0).split(",").length

    for (i <- 0 to numRows-1){
      var testLength = stringValues(i).split(",").length
      if (testLength != length){
        return false
      }
    }
    return true
  }

  def removeSqBrackets(df:DataFrame, spark:SparkSession):DataFrame ={

    //Remove "[" and "]" characters from string type columns
    var dfNew = df

    for (column <- df.columns){
      if (df.select(column).schema.fields(0).dataType.toString == "StringType"){
        dfNew = dfNew.withColumn(column.toString, regexp_replace(col(column.toString), "\\[", ""))
        dfNew = dfNew.withColumn(column.toString,regexp_replace(col(column.toString), "\\]", ""))
      }
    }
    return dfNew
  }

  def nonuniformHelp(values:String):String ={

    //Calculate mean value as string type, using string of comma-separated values
    if (values == null){
      return null
    } else{
      var newString = values.split(",").map(_.toDouble)
      var mean = newString.sum / newString.length
      return mean.toString
    }

  }

  def handleArrays(df:DataFrame, spark:SparkSession):DataFrame ={

    //Replaces columns with uniform number of values per example with each value as a separate column
    //Replaces columns with non-uniform number of values per example with its mean value
    //Columns with multiple values must be string type and comma-separated, as loaded by loadCSV
    var dfNew = df.select()

    for (column <- df.columns){
      if (df.select(column).schema.fields(0).dataType.toString == "StringType"){
        if (lengthUniform(df.select(column).filter(col(column).isNotNull), spark)){

          //Separate each value into separate column
          var firstExample = df.select(column).filter(col(column).isNotNull).take(1).mkString("")
          var nSplits = firstExample.split(",").length
          var dfSplit = df.select(column).withColumn("temp", split(col(column), ","))
            .select((0 until nSplits).map(i => col("temp").getItem(i).as(column.toString + i.toString)): _*)

          dfNew = mergeDataFrames(dfNew, dfSplit, spark)

        }else{

          //Replace by mean value
          val myUDF = udf(nonuniformHelp _)
          var dfTemp = df.select(column)
          dfTemp = dfTemp.withColumn(column, myUDF(df(column)))

          dfNew = mergeDataFrames(dfNew, dfTemp, spark)
        }
      } else {
        dfNew = mergeDataFrames(dfNew, df.select(column), spark)
      }
    }

    return dfNew
  }

  def convToDouble(df:DataFrame, spark:SparkSession):DataFrame ={

    //Cast all columns in dataframe as type Double
    var dfNew = df
    dfNew = dfNew.select(dfNew.columns.map(c => col(c).cast("double")): _*)

    dfNew
  }

  def impute(df:DataFrame, spark:SparkSession):DataFrame ={

    //Impute missing values with column mean
    var dfNew = convToDouble(df, spark)

    for (column <- dfNew.columns){

      //Calculate Column Mean
      var dfTemp = df.select(column).filter(col(column).isNotNull)
      var colMean = dfTemp.select(mean(dfTemp(column))).collect()
      var meanAsString = colMean(0)(0).toString

      //Impute nulls using mean
      dfNew = dfNew.withColumn(column, when(dfNew(column).isNull, lit(colMean(0)(0))).otherwise(dfNew(column)))
    }
    return dfNew
  }

}