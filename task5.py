import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, avg, window, hour, minute
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel

# STEP 1: Create Spark Session
spark = SparkSession.builder \
    .appName("Task5_FareTrendPrediction") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Paths
MODEL_PATH = "models/fare_trend_model_v2"
TRAINING_DATA_PATH = "training-dataset.csv"

# PART 1: OFFLINE MODEL TRAINING
if not os.path.exists(MODEL_PATH):
    print(f"\n[Training Phase] Training new model using {TRAINING_DATA_PATH}...")

    # STEP 2: Load training data
    hist_df_raw = spark.read.csv(TRAINING_DATA_PATH, header=True, inferSchema=False)

    # STEP 3: Cast columns
    hist_df_processed = hist_df_raw \
        .withColumn("event_time", col("timestamp").cast(TimestampType())) \
        .withColumn("fare_amount", col("fare_amount").cast(DoubleType()))

    # STEP 4: Aggregate into 5-minute windows
    hist_windowed_df = hist_df_processed \
        .groupBy(window(col("event_time"), "5 minutes")) \
        .agg(avg("fare_amount").alias("avg_fare"))

    # STEP 5: Engineer time-based features
    hist_features = hist_windowed_df \
        .withColumn("hour_of_day",    hour(col("window.start"))) \
        .withColumn("minute_of_hour", minute(col("window.start")))

    # STEP 6: Create VectorAssembler
    assembler = VectorAssembler(
        inputCols=["hour_of_day", "minute_of_hour"],
        outputCol="features"
    )
    train_df = assembler.transform(hist_features)

    # STEP 7: Create and train LinearRegression model
    lr = LinearRegression(
        featuresCol="features",
        labelCol="avg_fare"
    )
    model = lr.fit(train_df)

    # STEP 8: Save the model
    os.makedirs("models", exist_ok=True)
    model.write().overwrite().save(MODEL_PATH)
    print(f"[Model Saved] -> {MODEL_PATH}")

else:
    print(f"[Model Found] Using existing model at {MODEL_PATH}")


# PART 2: STREAMING INFERENCE
print("\n[Inference Phase] Starting real-time trend prediction stream...")

# STEP 9: Define schema
schema = StructType([
    StructField("trip_id",      StringType(),  True),
    StructField("driver_id",    IntegerType(), True),
    StructField("distance_km",  DoubleType(),  True),
    StructField("fare_amount",  DoubleType(),  True),
    StructField("timestamp",    StringType(),  True)
])

# STEP 10: Read from socket and parse data
raw_stream = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

parsed_stream = raw_stream \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*") \
    .withColumn("event_time", col("timestamp").cast(TimestampType()))

# STEP 11: Add watermark
parsed_stream = parsed_stream.withWatermark("event_time", "1 minute")

# STEP 12: Apply 5-minute windowed aggregation
windowed_df = parsed_stream \
    .groupBy(window(col("event_time"), "5 minutes", "1 minute")) \
    .agg(avg("fare_amount").alias("avg_fare"))

# STEP 13: Apply feature engineering
windowed_features = windowed_df \
    .withColumn("hour_of_day",    hour(col("window.start"))) \
    .withColumn("minute_of_hour", minute(col("window.start")))

# STEP 14: Apply VectorAssembler
assembler_inference = VectorAssembler(
    inputCols=["hour_of_day", "minute_of_hour"],
    outputCol="features"
)
feature_df = assembler_inference.transform(windowed_features)

# STEP 15: Load the pre-trained model
trend_model = LinearRegressionModel.load(MODEL_PATH)

# STEP 16: Make predictions
predictions = trend_model.transform(feature_df)

# STEP 17: Select final columns
output_df = predictions.select(
    col("window.start").alias("window_start"),
    col("window.end").alias("window_end"),
    "avg_fare",
    col("prediction").alias("predicted_next_avg_fare")
)

# STEP 18: Write to console
query = output_df.writeStream \
    .format("console") \
    .outputMode("append") \
    .option("truncate", False) \
    .start()

query.awaitTermination()