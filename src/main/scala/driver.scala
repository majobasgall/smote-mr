import java.time.{Duration, Instant}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import utils._


/**
  * SMOTE-MR: A distributed Synthetic Minority Oversampling Technique (SMOTE) for Big Data which applies a MapReduce based-approach. SMOTE-MR is classified as an `approximated/ non exact` solution, and there is an exact solution called SMOTE-BD written by the author (See: https://github.com/majobasgall/smote-bd)
  *
  * @author Maria Jose Basgall - @mjbasgall
  */


object driver {

  def main(args: Array[String]) {

    if (args.length < 12) {
      System.err.println("Wrong number of parameters\n\t")
      System.exit(1)
    }

    // Disabling "INFO" level logs (these lines must be before to create the SparkContext)
    Logger.getRootLogger.setLevel(Level.ERROR)

    //Getting parameters from spark-submit
    val parameters = args.map { arg =>
      arg.dropWhile(_ == '-').split('=') match {
        case Array(param, value) => param -> value
        case Array(param) => param -> ""
        case _ => throw new IllegalArgumentException("Invalid argument: " + arg)
      }
    }.toMap

    // Data parameters:
    val headerFile = parameters("headerFile")
    val inputFile = parameters("inputFile")
    val delimiter = parameters.getOrElse("delimiter", ", ")
    val outputPah = parameters("outputPah")


    // Algorithm parameters:
    val seed = parameters.getOrElse("seed", "1286082570").toInt
    val overPercentage = parameters.getOrElse("overPercentage", "100").toInt // > 0
    if (overPercentage <= 0) {
      System.err.println("The oversampling percentage must be greater than 0.\nYour value is\t: " + overPercentage)
      System.exit(1)
    }
    val k = parameters.getOrElse("K", "5").toInt
    val numPartitions = parameters.getOrElse("numPartitions", "20").toInt
    val minClassName = parameters.getOrElse("minClassName", "positive")

    // Create a SparkSession. No need to create SparkContext
    val spark = SparkSession
      .builder()
      .appName("SMOTE-MR")
      .getOrCreate()

    //set new options
    spark.conf.set("spark.ui.showConsoleProgress", "false")
    spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    spark.conf.set("spark.kryo.registrationRequired", "true")


    // Parse the header file to an array of maps, which will be used to transform the data into LabeledPoints (Spark API).
    val typeConversion = KeelParser.getParserFromHeader(spark.sparkContext, headerFile)
    val classes = typeConversion.apply(typeConversion.length - 1)
    //println("Classes information =\t" + classes)

    // Run SMOTE
    val before = Instant.now

    smote.runSMOTE_MR(spark, inputFile, delimiter, k, numPartitions, typeConversion, outputPah, seed, classes, minClassName, overPercentage)

    val after = Instant.now
    val delta = Duration.between(before, after).toMillis
    val deltaMin = Duration.between(before, after).toMinutes
    println("The algorithm SMOTE-MR has finished running. Time by Instant: " + delta + " ms\t (" + deltaMin + " min.) \n")
    spark.sparkContext.stop()
  }
}
