# Bucket S3 para Armazenamento
resource "aws_s3_bucket" "bucket" {
  bucket = "my-firehose-bucket"

  tags = {
    Name        = "My Firehose Bucket"
    Environment = "Dev"
  }
}

# Configuração do Kinesis Firehose
resource "aws_kinesis_firehose_delivery_stream" "firehose_stream" {
  name        = "btc_stream"
  destination = "extended_s3"

  extended_s3_configuration {
    role_arn   = "arn:aws:iam::471112828244:role/LabRole" # Verifique as permissões do role
    bucket_arn = aws_s3_bucket.bucket.arn

    # Buffering Configuration
    buffering_size     = 1   # Tamanho em MB
    buffering_interval = 60  # Intervalo em segundos

    compression_format = "GZIP" # Comprime os dados para reduzir custos de armazenamento

    # Prefixos para organização no bucket
    prefix              = "data/year=!{timestamp:yyyy}/month=!{timestamp:MM}/day=!{timestamp:dd}/hour=!{timestamp:HH}/"
    error_output_prefix = "errors/year=!{timestamp:yyyy}/month=!{timestamp:MM}/day=!{timestamp:dd}/hour=!{timestamp:HH}/!{firehose:error-output-type}/"

    # Configuração de Processamento
    processing_configuration {
      enabled = true

      # Adiciona delimitador ao final dos registros
      processors {
        type = "AppendDelimiterToRecord"
      }
    }

    # Configuração de Logs no CloudWatch
    cloudwatch_logging_options {
      enabled         = true
      log_group_name  = "/aws/kinesisfirehose/btc_stream"
      log_stream_name = "s3_delivery"
    }
  }

  # Tags para rastreamento
  tags = {
    Environment = "Dev"
    Team        = "Data"
  }
}
