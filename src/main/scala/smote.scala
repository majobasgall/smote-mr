import java.time.{Duration, Instant}

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.feature.{LabeledPoint, MinMaxScaler}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import utils._

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

/**
  * SMOTE-MR: A distributed Synthetic Minority Oversampling Technique (SMOTE) for Big Data which applies a MapReduce based-approach. SMOTE-MR is classified as an `approximated/ non exact` solution, and there is an exact solution called SMOTE-BD written by the author (See: https://github.com/majobasgall/smote-bd)
  *
  * @author Maria Jose Basgall - @mjbasgall
  */


object smote {

  type InstanceKey = Long
  type InstanceKeyLp = (InstanceKey, LabeledPoint)
  type KNeighborsKeys = Array[InstanceKey]
  type InstanceAndItsNeighbors = (InstanceKey, KNeighborsKeys)

  val Preserves = true //To use with ``mapPartitions`` to preserve the index partitions

  /**
    *
    * @param sc            SparkContext
    * @param inPath        The complete input file path
    * @param delimiter     The character which delimits each feature
    * @param k             How many neighbours to get
    * @param numPartitions The number of maps to use
    * @param bcTypeConv    The parsed header file
    * @param outPath       output path
    * @param seed          to take samples
    */
  def runSMOTE_MR(sc: SparkSession,
                  inPath: String,
                  delimiter: String,
                  k: Int,
                  numPartitions: Int,
                  bcTypeConv: Array[Map[String, Double]],
                  outPath: String,
                  seed: Int,
                  classes: Map[String, Double],
                  minClassName: String,
                  overPerc: Int): Unit = {

    val minClassNumber = classes.apply(minClassName)
    val majClassNumber = classes.find(_._1 != minClassName).get._2
    val majClassName = classes.find(_._1 != minClassName).get._1

    println("Minority class name:\t" + minClassName + "\t/ Keel number:\t" + minClassNumber)
    println("Majority class name:\t" + majClassName + "\t/ Keel number:\t" + majClassNumber)

    /*  Get each point of the input file as a LabeledPoint: [double, Vector[double]].
        For instance: (0.0,[0.0,0.56,0.45,0.185,1.07,0.3805,0.175,0.41]) */

    // `numPartitions` is a suggested minimum number of partitions for the resulting RDD
    val allData = sc.sparkContext.textFile(inPath: String, numPartitions).filter(line => !line.isEmpty && line.split(",").length == bcTypeConv.length)
      .map(line => KeelParser.parseLabeledPoint(bcTypeConv, line))
    // .map(e => { LabeledPoint(e.label, Vectors.dense(e.features.toArray.slice(0,500)))  } )   ONLY FOR TESTING!

    val numAll = allData.count()
    println("Original training dataset size\t" + numAll)

    /*
      MinMaxScaler - Data normalization
     */
    // The following import doesn't work externally because the implicits object is defined inside the SQLContext class
    import sc.implicits._

    // Create a DataFrame from the RDD[LabeledPoint], with the following two columns:
    //|labels|            features|
    //+------+--------------------+
    //|   1.0|[6.0,7.0,42.0,1.1...|
    val allDF = allData.map(e => (e.label, e.features)).toDF("labels", "features")
    allDF.persist() // it'll be used more than 1 time in the future
    //allDF.show(5,false)

    // MaxMin Scaler with the min(0) and max(1)
    val scaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("featuresScaled")
      .setMax(1)
    // .setMin(0)

    // Scaling and drop the `features` column
    /*  +------+--------------------+--------------------+
        |labels|            features|      featuresScaled|
        +------+--------------------+--------------------+
        |   1.0|[6.0,7.0,42.0,1.1...|[1.0,0.2142857142...|*/
    val fittedDF = scaler.fit(allDF)
    val vector_original_min = fittedDF.originalMin
    val vector_original_max = fittedDF.originalMax

    // Apply transforming only the minority data (but using the whole statistic of allDF)
    // Drop the features column and adding a column with an id
    val normalizedDF = fittedDF.transform(allDF.filter($"labels" === minClassNumber))
      .drop("features").withColumn("idx", monotonically_increasing_id())

    // Convert from the DataFrame to RDD[LabeledPoint]
    // After normalization, apply repartition to put data across the `numPartitions` partitions (because in the reading file part the number of partitions is taken only as suggestion)
    val posData = normalizedDF.rdd.map {
      row =>
        (row.getAs[Long]("idx"),
          LabeledPoint(
            row.getAs[Double]("labels"),
            row.getAs[Vector]("featuresScaled")
          )
        )
    }.repartition(numPartitions)

    allDF.unpersist() // unpersist because it wont be used anymore
    posData.persist()
    // posData.persist(StorageLevels.MEMORY_AND_DISK_SER)


    val numPos = posData.count()
    val numNeg = numAll - numPos
    println("MINORITY instances number (" + minClassName + ") :\t" + numPos)
    println("MAJORITY instances number (" + majClassName + "):\t" + numNeg)

    if (numPos <= k) {
      System.err.println("Positive instances must be greater than 'k' value")
      System.exit(1)
    }
    if ((numPos / numPartitions.toDouble) < (k + 1)) {
      System.err.println("There are no sufficient instances to work with the number of partitions chosen")
      System.exit(1)
    }

    /*
    The minority class will be modified an `overPerc` % comparing with the num of elements of the majority class.

    Thus, for instance, if numNeg = 500 and numPos = 50,
    If overPerc = 100   (1.0x) =>  finalQtyPos = 500 and numSyn = 450 new instances. Both the same quantity of intances.
    If overPerc = 50    (0.5x) =>  finalQtyPos = 250 and numSyn = 200 new instances. MinClass will have half number of elem than MajClass.
    If overPerc = 150   (1.5x) =>  finalQtyPos = 750 and numSyn = 700 new instances.
    If overPerc = 200   (2.0x) =>  finalQtyPos = 1000 and numSyn = 950 new instances. MinClass will have twice as many elements than MajClass.
     */
    val finalQtyPos = math.ceil(numNeg * overPerc / 100).toInt //round up
    val numSyn = finalQtyPos - numPos
    val negPerc = numNeg * 100 / numAll
    val posPerc = numPos * 100 / numAll
    println("Current classes distribution\t NEG:\t" + negPerc + " %\t- POS:\t " + posPerc + " %")
    //println("Desired oversampling percentage (to apply on the minority class): \t" + overPerc + " %.")
    println("EXACT TOTAL MINORITY instances (orig and synth):\t" + finalQtyPos)
    println("HAS TO BE CREATED:\t" + numSyn)
    var creationFactor = math.ceil(numSyn / numPos).toInt //round up


    // Defining an acceptable range for the final minority number of instances:
    val low_limit = (numSyn * (overPerc - 1)) / overPerc
    val up_limit = (numSyn * (overPerc + 1)) / overPerc
    println("Acceptable range:\t[" + low_limit + "\t;\t" + up_limit + "]")
    val qty_syn = creationFactor * numPos

    // checking if the creation factor of artificial instances will generate an acceptable value of instances
    // decides if will be necessary to cut the artificial RDD
    if ((low_limit < qty_syn) && (up_limit > qty_syn)) {
      println("It is not necessary to increase the `creationFactor` value")
    } else creationFactor += 1

    println("Each current minority instance has to create \t" + creationFactor + " artificial points.")

    //println("INFO FOR posData\t" + printRDDInfo(posData))

    /*
    kNN stage
     */
    val beforeKnn = Instant.now
    val knnPerPart = posData.mapPartitionsWithIndex(
      (_, partData) => {
        val (keepIt, breakItDown) = partData.duplicate
        val nn = new ListBuffer[(Long, Array[Long])]
        val keepArrBuf = keepIt.to[ArrayBuffer]

        //For each instance, calculates the knn:
        breakItDown.foreach { curInstance => nn += localKNearestNeighbors(curInstance, keepArrBuf, k) }
        nn.toIterator
      }
      , preservesPartitioning = Preserves
    )

    val afterKnn = Instant.now
    val deltaKnn = Duration.between(beforeKnn, afterKnn).toMillis
    println("Time for the kNN stage\t" + deltaKnn + " ms")


    /* Link the neighbours (KNeighborsKeys) with the trainData (LabeledPoint), joining them using the `InstanceKey`.
     * After that, it is converted to a map: <InstanceKey, (KNeighborsKeys, LabeledPoint)>  */
    val beforeJoin = Instant.now
    val gNN = knnPerPart.join(posData).collectAsMap()
    val afterJoin = Instant.now
    val deltaJoin = Duration.between(beforeJoin, afterJoin).toMillis
    println("Time for the joining data stage\t" + deltaJoin + " ms\n")

    var newInstance = ""
    var artificialData = ArrayBuffer[String]() // ArrayBuffer is a mutable data structure which allows to access and modify elements at specific index
    val rand = new scala.util.Random(seed)

    /*
    SMOTE stage
     */
    val beforeSMT = Instant.now
    val synData = posData.mapPartitionsWithIndex(
      (_, dataPart) => {
        var curKey = 0L
        var neighList: KNeighborsKeys = null
        var neighKey = 0L
        var neighLp: LabeledPoint = null
        var control = false

        dataPart.foreach {
          curInstance => {
            curKey = curInstance._1
            neighList = gNN.apply(curKey)._1
            for {
              num_created <- 0 until creationFactor
              _ = {
                neighKey = neighList(rand.nextInt(k))
                neighLp = gNN.apply(neighKey)._2
                newInstance = interpolation(curInstance._2.features, neighLp.features).toString
                artificialData.+=(newInstance.substring(1, newInstance.length() - 1) + delimiter + minClassName)
              }
            } yield ()
          }
        }
        artificialData.toIterator
      }
      , preservesPartitioning = Preserves
    )
    val afterSMT = Instant.now
    val deltaSMT = Duration.between(beforeSMT, afterSMT).toMillis
    println("Time for the SMT stage\t" + deltaSMT + "\tms")

    val numCreated = synData.count()
    println("Number of created instances\t" + numCreated)
    var cut = false
    if (numCreated < low_limit && numCreated > up_limit) {
      println("It has to be cut the artificial instances number")
      cut = true
    } else println("The number of artificial instances is on the acceptable range")


    /*
    Cutting synData
     */
    var cutSynData = sc.sparkContext.emptyRDD[String] //Get an RDD of String type that has no partitions or elements
    if (cut) {
      val beforeTake = Instant.now

      //take only the exact number and make it RDD again
      val zipped = synData.zipWithIndex()
      cutSynData = zipped.filter(_._2 < numSyn).keys

      val afterTake = Instant.now
      val deltaTake = Duration.between(beforeTake, afterTake).toMillis
      println("Time for the cutting stage\t" + deltaTake + "\tms")
    } else {
      println("Time for the cutting stage\tN/A")
    }


    /**
      * Data de-normalization
      */
    val beforeDenorm = Instant.now
    // total number of columns (features + class)
    val numCols = bcTypeConv.length
    val selectCols = (0 until (numCols - 1)).map(i => $"arr" (i).as(s"col_$i"))

    /* convert the RDD[String] to RDD[Array[String]], and take off the labels of the converted DF */
    val synDataNoLabelsDF = if (cut) cutSynData.map(e => e.split(",")).toDF("arr").select(selectCols: _*)
    else synData.map(e => e.split(",")).toDF("arr").select(selectCols: _*)
    synDataNoLabelsDF.persist() // 2 times

    val updateFunction = (columnValue: Column, minValue: Double, maxValue: Double) =>
      (columnValue * (lit(maxValue) - lit(minValue))) + lit(minValue)

    val updateColumns = (df: DataFrame, minVector: Vector, maxVector: Vector, updateFunction: (Column, Double, Double) => Column) => {
      val columns = df.columns
      minVector.toArray.zipWithIndex.map {
        case (_, index) => updateFunction(col(columns(index)), minVector(index), maxVector(index)).as(columns(index))
      }
    }

    // Applying de-normalization and adding `minClassName` class column
    val denormdDF = synDataNoLabelsDF.select(
      updateColumns(synDataNoLabelsDF, vector_original_min, vector_original_max, updateFunction): _*
    ).withColumn("col_" + numCols, lit(minClassName))


    // Convert Dataframe to RDD[String]
    val synRDD = denormdDF.rdd.map(_.mkString(", "))

    val afterDenorm = Instant.now
    val deltaDenorm = Duration.between(beforeDenorm, afterDenorm).toMillis
    println("Time for the denormalization stage\t" + deltaDenorm + "\tms")

    synDataNoLabelsDF.unpersist()
    posData.unpersist()


    /* Convert RDD[LabeledPoint] to RDD[String] */
    val originalData = allData.map(x => x.features.toArray.mkString(delimiter) + delimiter + (if (x.label == minClassNumber) minClassName else majClassName)) //.cache()


    /* Save results */
    val fs = FileSystem.get(sc.sparkContext.hadoopConfiguration)
    if (fs.exists(new Path(outPath))) {
      println("Output directory already exists. Deleting...")
      fs.delete(new Path(outPath), true)
    }

    val outputData = synRDD.union(originalData)
    outputData.saveAsTextFile(outPath, classOf[org.apache.hadoop.io.compress.GzipCodec]) // save results in compressed part-... files
    println("Results have been saved on\t" + outPath)

    val numSynRDD = synRDD.count()
    val posPercFinal = ((numPos + numSynRDD) * 100) / numNeg

    println("FINAL classes distribution\t NEG:\t100 %\t- POS:\t " + posPercFinal + " %")
    println("FINAL classes quantities\t NEG:\t" + numNeg + " \t- POS:\t " + (numPos + numSynRDD) + " ")

    sc.close()
  }


  /**
    * Calculates the kNN for an instance
    *
    * @param x     an instance
    * @param train a set of instances
    * @param k     number of nearest neighbors
    * @return
    */
  def localKNearestNeighbors(x: InstanceKeyLp, train: ArrayBuffer[InstanceKeyLp], k: Int): (Long, Array[Long]) = {
    // def localKNearestNeighbors(x: InstanceKeyLp, train: ArrayBuffer[InstanceKeyLp], k: Int): (Long, Array[Long], Float) = {
    var dist = 0.0f
    val nearest = Array.fill(k)(-1)
    val distA = Array.fill(k)(0.0f)
    val size = train.length

    for (i <- 0 until size) {
      dist = euclideanDist(x._2.features, train(i)._2.features)
      if (dist > 0.0f) { //leave-one-out control
        var stop = false
        var j = 0
        while (j < k && !stop) { //Check if it can be inserted as NN
          if (nearest(j) == (-1) || dist <= distA(j)) { // if `nearest(j) is empty or the current dist is less than `distA(j)`, insert a new neighbor!
            // displacement of the elements to the right
            for (l <- ((j + 1) until k).reverse) { //for (int l = k - 1; l >= j + 1; l--)
              nearest(l) = nearest(l - 1)
              distA(l) = distA(l - 1)
            }

            nearest(j) = i
            distA(j) = dist
            stop = true
          }
          j += 1
        }
      }
    }

    val key = x._1
    val neighbors = new Array[InstanceKey](k)
    var meanDist = 0f

    for (i <- 0 until k) {
      neighbors(i) = train(nearest(i))._1
      meanDist = meanDist + distA(i)
    }

    meanDist = meanDist / k

    //(key, neighbors, meanDist)
    (key, neighbors)
  }

  /** Computes the Euclidean distance between instance x and instance y.
    * The type of the distance used is determined by the value of distanceType.
    *
    * @param x instance x
    * @param y instance y
    * @return Euclidean distance
    */
  def euclideanDist(x: Vector, y: Vector): Float = {
    var sum = 0.0
    val size = x.size

    for (i <- 0 until size) sum += (x(i) - y(i)) * (x(i) - y(i))

    Math.sqrt(sum).toFloat
  }

  /**
    * Create an artificial instance
    *
    * @param sf sampleFeatures (labeledPoint Vector)
    * @param nf neighbourFeatures (labeledPoint Vector)
    * @return the new features vector for the artificial instance
    */
  private def interpolation(sf: Vector, nf: Vector): Vector = {
    val size = sf.size
    val rand = new scala.util.Random().nextDouble()
    val result = new Array[Double](size)
    var difference = 0.0

    for (i <- 0 until size) {
      difference = (nf(i) - sf(i)) * rand
      result(i) = sf(i) + difference
    }

    Vectors.dense(result).compressed
  }

  /**
    *
    * @param anRdd to print its information
    * @tparam A represents any type of data
    * @return
    */
  def printRDDInfo[A](anRdd: RDD[A]): String = {
    // Get info of each partition
    //`.glom()` applies coalesce and return as an array
    val info = anRdd.glom().map(_.length).collect()
    val avg = info.sum / info.length
    val str = "Min:\t" + info.min + " - Max:\t" + info.max + " - avg:\t " + avg + " - numParts:\t" + info.length
    str
  }

  /**
    *
    * @param anRdd to print its content
    * @param n     the number of elements to show
    * @tparam A represents any type of data
    */
  def printRddContent[A](anRdd: RDD[A], n: Int): Unit = {
    anRdd match {
      case r3: RDD[(Long, Array[Long], Float)] => r3.asInstanceOf[RDD[(Long, Array[Long], Float)]].map { case (key, arr, dist) => (key, arr.toString, dist) }.take(n).foreach(println)
      case r4: RDD[(Long, Array[Long])] => r4.asInstanceOf[RDD[(Long, Array[Long])]].map { case (key, arr) => (key, arr.toString) }.take(n).foreach(println)
      case r1: RDD[InstanceAndItsNeighbors] => r1.asInstanceOf[RDD[InstanceAndItsNeighbors]].map { case (a, arr) => (a, arr.toList) }.take(n).foreach(println)
      case r2: RDD[(InstanceKey, LabeledPoint)] => r2.asInstanceOf[RDD[(InstanceKey, LabeledPoint)]].map { case (a, arr) => (a, arr.toString()) }.take(n).foreach(println)
      case _ => anRdd.foreach(println)
    }
    println()
  }
}
