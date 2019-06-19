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
from pyspark.sql.types import DoubleType,IntegerType
from pyspark.sql.functions import when
import pyspark.sql.functions as F
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import Binarizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer,OneHotEncoderEstimator,Binarizer
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.mllib.evaluation import BinaryClassificationMetrics

#load data and remove '?'
data = spark.read.csv("../Data/train_set.csv",header="true")
data = data.select('Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8','NVVar1','NVVar2','NVVar3','NVVar4',
                  'Cat1','Cat3','Cat6','Cat8','Cat9','Cat10','Cat11','Cat12','Calendar_Year','Model_Year','Claim_Amount')
for col in data.columns:
    data = data.filter((data[col] != '?'))

#transfer data type 
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

#transfer categorical value to index
categorical = {'Cat1':11,'Cat3':7,'Cat6':7,'Cat8':4,'Cat9':2,'Cat10':4,'Cat11':7,'Cat12':7}
for col,num in categorical.items():
    name = col+'_index'
    indexer = StringIndexer(inputCol=col, outputCol=name)
    data = indexer.fit(data).transform(data)

#OneHotEncoderEstimator for catogories representation
data = data.select('Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8','NVVar1','NVVar2','NVVar3','NVVar4',
                  'Cat1_index','Cat3_index','Cat6_index','Cat8_index','Cat9_index','Cat10_index','Cat11_index',
                   'Cat12_index','Calendar_Year','Model_Year','Claim_Amount')
category = ['Cat1_index','Cat3_index','Cat6_index','Cat8_index','Cat9_index','Cat10_index','Cat11_index','Cat12_index']
new_cat = []
for col in category:
    name = col.replace('_index','_vec')
    new_cat.append(name)
    num_column.append(name)
encoder = OneHotEncoderEstimator(inputCols=category,outputCols=new_cat)
one_model = encoder.fit(data)
data = one_model.transform(data)

#question 1.3 handle the unbalanced data
data = data.withColumn('weight',when((data['Claim_Amount'] != 0), 0.99).otherwise(0.01))
#allocate label for each row
data = data.withColumn('type',when((data['Claim_Amount'] != 0), 1).otherwise(0))


#feature choose and PCA
data = data.select('Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8','NVVar1','NVVar2','NVVar3','NVVar4',
                  'Cat1_vec','Cat3_vec','Cat6_vec','Cat8_vec','Cat9_vec','Cat10_vec','Cat11_vec','Cat12_vec',
                   'Calendar_Year','Model_Year','Claim_Amount','weight','type')
data = data.withColumnRenamed('type','label')
assembler = VectorAssembler(inputCols = num_column, outputCol = 'old_features')
data = assembler.transform(data)
pca = PCA(k=30, inputCol='old_features', outputCol="features")
pca_model = pca.fit(data)
data = pca_model.transform(data)
result = data.select('features','label','weight','Claim_Amount')
(trainingData, testData) = result.randomSplit([0.7, 0.3], 66)

#classification
lr = LogisticRegression(labelCol ='label',weightCol = 'weight')
logi_model = lr.fit(trainingData)
logi_prediction = logi_model.transform(testData)

#show the output of classification
logi_prediction.groupBy('label','prediction').count().show()

#build the gamma regression model
non_zero1 = trainingData.filter((trainingData['label'] != 0))
non_zero2 = testData.filter((testData['label'] != 0))
glm_poisson = GeneralizedLinearRegression(featuresCol='features', labelCol='Claim_Amount', maxIter=10, regParam=0.01,\
                                          family='Gamma', link='identity')
glm_model = glm_poisson.fit(non_zero1)

#combine the output of linear regression and classification
pred_zero = logi_prediction.filter(logi_prediction['prediction']==0)
pred_zero = pred_zero.withColumn('claim_prediction',pred_zero['label']*0).select('Claim_Amount','claim_prediction')
pred_nonzero = logi_prediction.filter(logi_prediction['prediction']!=0)
pred_nonzero = pred_nonzero.select('features','Claim_Amount')
pred_amount = glm_model.transform(pred_nonzero)
pred_amount = pred_amount.select('Claim_Amount','prediction')
pred_amount = pred_amount.withColumnRenamed('prediction','claim_prediction')
q2_result = pred_amount.union(pred_zero)

def performance(prediction):
    '''
    performance of model
    '''
    binarizer = Binarizer(threshold=0.5, inputCol="claim_prediction", outputCol="prediction")
    binarizedDataFrame = binarizer.transform(prediction)
    binarizer = Binarizer(threshold=0.5, inputCol="Claim_Amount", outputCol="label")
    binarizedDataFrame = binarizer.transform(binarizedDataFrame)
    prediction_label = binarizedDataFrame.select('prediction','label')
    metrics = BinaryClassificationMetrics(prediction_label.rdd)
    return metrics.areaUnderROC
q2_result = q2_result.withColumn('Claim_Amount', q2_result['Claim_Amount'].cast(DoubleType()))
ROC = performance(q2_result)
print("the area under ROC is:",round(ROC,2))

evaluator = RegressionEvaluator\
      (labelCol="Claim_Amount", predictionCol="claim_prediction", metricName="rmse")
glm_rmse = evaluator.evaluate(q2_result)
print("The RMSE of my gamma regression", glm_rmse)

spark.stop()
