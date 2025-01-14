resource "aws_s3_bucket" "fiap_etl_bucket" {
  bucket = "fiap-etl-efog"
}

resource "aws_s3_object" "raw_path" {
  bucket = aws_s3_bucket.fiap_etl_bucket.bucket
  key    = "raw/"
}

resource "aws_s3_object" "interim_path" {
  bucket = aws_s3_bucket.fiap_etl_bucket.bucket
  key    = "interim/"
}

resource "aws_s3_object" "final_path" {
  bucket = aws_s3_bucket.fiap_etl_bucket.bucket
  key    = "final/"
}

resource "aws_s3_object" "query_results_path" {
  bucket = aws_s3_bucket.fiap_etl_bucket.bucket
  key    = "query-results/"
}

resource "aws_s3_object" "model_metrics" {
  bucket = aws_s3_bucket.fiap_etl_bucket.bucket
  key    = "model-metrics/"
}
