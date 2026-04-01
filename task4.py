import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, abs as abs_diff
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel

# STEP 1: Create Spark Session
spark = SparkSession.builder \
    .appName("Task4_FarePrediction") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Paths
MODEL_PATH = "models/fare_model"
TRAINING_DATA_PATH = "training-dataset.csv"

# PART 1: OFFLINE MODEL TRAINING
if not os.path.exists(MODEL_PATH):
    print(f"\n[Training Phase] No model found. Training a new model using {TRAINING_DATA_PATH}...")

    # STEP 2: Load training data
    train_df_raw = spark.read.csv(TRAINING_DATA_PATH, header=True, inferSchema=False)

    # STEP 3: Cast columns to DoubleType
    train_df = train_df_raw \
        .withColumn("distance_km", col("distance_km").cast(DoubleType())) \
        .withColumn("fare_amount", col("fare_amount").cast(DoubleType()))

    # STEP 4: Create VectorAssembler
    assembler = VectorAssembler(
        inputCols=["distance_km"],
        outputCol="features"
    )
    train_data_with_features = assembler.transform(train_df)

    # STEP 5: Create LinearRegression model
    lr = LinearRegression(
        featuresCol="features",
        labelCol="fare_amount"
    )

    # STEP 6: Train the model
    model = lr.fit(train_data_with_features)

    # STEP 7: Save the model
    os.makedirs("models", exist_ok=True)
    model.write().overwrite().save(MODEL_PATH)
    print(f"[Training Complete] Model saved to -> {MODEL_PATH}")

else:
    print(f"[Model Found] Using existing model from {MODEL_PATH}")


# PART 2: STREAMING INFERENCE
print("\n[Inference Phase] Starting real-time fare prediction stream...")

# STEP 8: Define schema
schema = StructType([
    StructField("trip_id",      StringType(),  True),
    StructField("driver_id",    IntegerType(), True),
    StructField("distance_km",  DoubleType(),  True),
    StructField("fare_amount",  DoubleType(),  True),
    StructField("timestamp",    StringType(),  True)
])

# STEP 9: Read streaming data from socket
raw_stream = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

# STEP 10: Parse JSON
parsed_stream = raw_stream \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# STEP 11: Load the pre-trained model
model = LinearRegressionModel.load(MODEL_PATH)

# STEP 12: Apply VectorAssembler to streaming data
assembler_inference = VectorAssembler(
    inputCols=["distance_km"],
    outputCol="features"
)
stream_with_features = assembler_inference.transform(parsed_stream)

# STEP 13: Make predictions
predictions = model.transform(stream_with_features)

# STEP 14: Calculate deviation between actual and predicted fare
predictions_with_deviation = predictions.withColumn(
    "deviation",
    abs_diff(col("fare_amount") - col("prediction"))
)

# STEP 15: Select final columns
output_df = predictions_with_deviation.select(
    "trip_id",
    "driver_id",
    "distance_km",
    "fare_amount",
    col("prediction").alias("predicted_fare"),
    "deviation"
)

# STEP 16: Write to console
query = output_df.writeStream \
    .format("console") \
    .outputMode("append") \
    .option("truncate", False) \
    .start()

query.awaitTermination()