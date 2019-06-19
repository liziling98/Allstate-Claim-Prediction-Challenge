import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Spark Intro") \
    .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

from pyspark.sql import Row,Column
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import when
import pyspark.sql.functions as F
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.feature import Binarizer

#preprocessing
data = spark.read.csv("../Data/train_set.csv",header="true")
data = data.select('Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8','NVVar1','NVVar2','NVVar3','NVVar4',
                  'Cat1','Cat3','Cat6','Cat8','Cat9','Cat10','Cat11','Cat12','Calendar_Year','Model_Year','Claim_Amount')
for col in data.columns:
    data = data.filter((data[col] != '?'))

num_column = ['Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8','NVVar1','NVVar2','NVVar3','NVVar4']
for col in num_column:
    data = data.withColumn(col, data[col].cast(DoubleType()))
num_column1 = ['Calendar_Year','Model_Year','Claim_Amount']
for col in num_column1:
    data = data.withColumn(col, data[col].cast(IntegerType()))

data = data.withColumn('Calendar_Year', (data['Calendar_Year']-2005))
num_column.append('Calendar_Year')

data = data.withColumn('Model_Year', (data['Model_Year']-1981))
num_column.append('Model_Year')

#pca choose features
categorical = {'Cat1':11,'Cat3':7,'Cat6':7,'Cat8':4,'Cat9':2,'Cat10':4,'Cat11':7,'Cat12':7}
from pyspark.ml.feature import StringIndexer
for col,num in categorical.items():
    name = col+'_index'
    indexer = StringIndexer(inputCol=col, outputCol=name)
    data = indexer.fit(data).transform(data)
data = data.select('Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8','NVVar1','NVVar2','NVVar3','NVVar4',
                  'Cat1_index','Cat3_index','Cat6_index','Cat8_index','Cat9_index','Cat10_index','Cat11_index',
                   'Cat12_index','Calendar_Year','Model_Year','Claim_Amount')
from pyspark.ml.feature import OneHotEncoderEstimator
category = ['Cat1_index','Cat3_index','Cat6_index',
                   'Cat8_index','Cat9_index','Cat10_index','Cat11_index','Cat12_index']
new_cat = []
for col in category:
    name = col.replace('_index','_vec')
    new_cat.append(name)
    num_column.append(name)
    
encoder = OneHotEncoderEstimator(inputCols=category,
                                 outputCols=new_cat)
one_model = encoder.fit(data)
data = one_model.transform(data)
data = data.select('Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8','NVVar1','NVVar2','NVVar3','NVVar4',
                  'Cat1_vec','Cat3_vec','Cat6_vec','Cat8_vec','Cat9_vec','Cat10_vec','Cat11_vec','Cat12_vec',
                   'Calendar_Year','Model_Year','Claim_Amount')
data = data.withColumnRenamed('Claim_Amount','label')
assembler = VectorAssembler(inputCols = num_column, outputCol = 'old_features')
data = assembler.transform(data)
pca = PCA(k=30, inputCol='old_features', outputCol="features")
pca_model = pca.fit(data)
data = pca_model.transform(data)

#train and test
result = data.select('features','label')
(trainingData, testData) = result.randomSplit([0.7, 0.3], 66)
non_zero = trainingData.filter((trainingData['label'] != 0))
is_zero = trainingData.filter((trainingData['label'] == 0))
train2 = non_zero.sample(True,25.0,66)
trainingData = train2.union(is_zero)

trainingData.cache()  
testData.cache()

#linear regression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
lr = LinearRegression(maxIter=10, regParam=0.05)
lrModel = lr.fit(trainingData)
lr_prediction = lrModel.transform(testData)
evaluator = RegressionEvaluator\
      (labelCol="label", predictionCol="prediction",metricName="rmse")
rmse = evaluator.evaluate(lr_prediction)
print("RMSE = ",rmse)

def performance(prediction):
    '''
    performance of model
    '''
    binarizer = Binarizer(threshold=0.5, inputCol="prediction", outputCol="b_prediction")
    binarizedDataFrame = binarizer.transform(prediction)
    binarizer = Binarizer(threshold=0.5, inputCol="label", outputCol="b_label")
    binarizedDataFrame = binarizer.transform(binarizedDataFrame)
    prediction_label = binarizedDataFrame.select('b_prediction','b_label')
    metrics = BinaryClassificationMetrics(prediction_label.rdd)
    return metrics.areaUnderROC
lr_prediction = lr_prediction.withColumn('label', lr_prediction['label'].cast(DoubleType()))
ROC = performance(lr_prediction)
print("ROC = ",ROC)

spark.stop()
