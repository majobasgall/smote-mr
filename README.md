# SMOTE-MR
SMOTE-MR: A distributed Synthetic Minority Oversampling Technique (SMOTE) [1] for Big Data which applies a MapReduce based-approach. SMOTE-MR is categorized as an `approximated/ non exact` solution. Also, there is an `exact` solution called SMOTE-BD written by the author (See: https://github.com/majobasgall/smote-bd)

## How to run it?

A generic example to run it could be:

```spark-submit --master "URL" --executor-memory "XG" "path-to-jar".jar --class "path-to-main" --datasetName="aName" --headerFile="path-to-header" --inputFile="path-to-input" --delimiter=", " --outputPah="path-to-output" --seed="aSeed" --K="number-of-neighbours" --numPartitions="number-of-parts"  --nReducers="number-of-reducers" --numIterations="number-of-iterations" --minClassName="min-class-name" -overPercentage=100 ```

- Parameters of spark: ```--master "URL" | --executor-memory "XG" ```. They can be useful for launch with diferent settings and datasets.
- ```--class path.to.the.main aJarFile.jar``` Determine the jar file to be run.
- ```datasetName``` The name of the current dataset.
- ```headerFile``` Full path to header file.
- ```inputFile``` Full path to input file.
- ```delimiter``` Delimiter of each attribute value.
- ```outputPah``` Full path to output directory.
- ```seed``` A seed to generate random numbers.
- ```K``` Number of nearest neighbours.
- ```numPartitions``` Number of partitions to split data.
- ```nReducers``` Number of reducers (required by the K-NN stage).
- ```numIterations``` Number of iterations (required by the K-NN stage).
- ```minClassName``` Name of the minority class (according to the header file).
- ```overPercentage``` Percentage of balancing between classes.

## References
[1] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. J. Artif. Int. Res., 16(1), 321–357.
