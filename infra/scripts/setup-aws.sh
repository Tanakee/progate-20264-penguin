#!/usr/bin/env bash
set -euo pipefail

# ────────────────────────────────────────────────────
# ファースト肯定ペンギン - AWS リソース一括構築
#
# 前提:
#   - AWS CLI v2 がインストール済み
#   - ~/.aws/credentials に有効な認証情報がある
#   - ap-northeast-1 リージョンを使用
#
# 使い方:
#   ./infra/scripts/setup-aws.sh
# ────────────────────────────────────────────────────

REGION="ap-northeast-1"
PROJECT="first-penguin"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
S3_BUCKET="${PROJECT}-bucket-${ACCOUNT_ID}"
DYNAMODB_TABLE="${PROJECT}-clap-events"
LAMBDA_NAME="${PROJECT}-processor"
APPSYNC_NAME="${PROJECT}-api"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== ファースト肯定ペンギン AWS セットアップ ==="
echo "Region: ${REGION}"
echo "Account: ${ACCOUNT_ID}"
echo ""

# ── 1. S3 バケット ──────────────────────────────────
echo "[1/7] S3 バケット..."
if aws s3api head-bucket --bucket "$S3_BUCKET" 2>/dev/null; then
    echo "  既に存在: ${S3_BUCKET}"
else
    aws s3api create-bucket \
        --bucket "$S3_BUCKET" \
        --region "$REGION" \
        --create-bucket-configuration LocationConstraint="$REGION" \
        --output text > /dev/null
    echo "  作成完了: ${S3_BUCKET}"
fi

# ── 2. DynamoDB テーブル ────────────────────────────
echo "[2/7] DynamoDB..."
if aws dynamodb describe-table --table-name "$DYNAMODB_TABLE" --region "$REGION" 2>/dev/null; then
    echo "  既に存在: ${DYNAMODB_TABLE}"
else
    aws dynamodb create-table \
        --table-name "$DYNAMODB_TABLE" \
        --attribute-definitions \
            AttributeName=id,AttributeType=S \
            AttributeName=timestamp,AttributeType=S \
        --key-schema \
            AttributeName=id,KeyType=HASH \
            AttributeName=timestamp,KeyType=RANGE \
        --billing-mode PAY_PER_REQUEST \
        --region "$REGION" \
        --output text > /dev/null
    echo "  作成完了: ${DYNAMODB_TABLE}"
    echo "  テーブル有効化を待機中..."
    aws dynamodb wait table-exists --table-name "$DYNAMODB_TABLE" --region "$REGION"
    echo "  有効化完了"
fi

# ── 3. IAM ロール (Lambda 用) ───────────────────────
echo "[3/7] IAM ロール..."

# Learner Lab では LabRole が既に存在するのでそれを使用
LAMBDA_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/LabRole"
if aws iam get-role --role-name "LabRole" 2>/dev/null; then
    echo "  LabRole を使用: ${LAMBDA_ROLE_ARN}"
else
    # LabRole がなければ作成
    LAMBDA_ROLE_NAME="${PROJECT}-lambda-role"
    ASSUME_ROLE_POLICY='{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }]
    }'

    if aws iam get-role --role-name "$LAMBDA_ROLE_NAME" 2>/dev/null; then
        LAMBDA_ROLE_ARN=$(aws iam get-role --role-name "$LAMBDA_ROLE_NAME" --query 'Role.Arn' --output text)
        echo "  既に存在: ${LAMBDA_ROLE_NAME}"
    else
        LAMBDA_ROLE_ARN=$(aws iam create-role \
            --role-name "$LAMBDA_ROLE_NAME" \
            --assume-role-policy-document "$ASSUME_ROLE_POLICY" \
            --query 'Role.Arn' --output text)
        echo "  作成完了: ${LAMBDA_ROLE_NAME}"

        POLICY_DOC=$(cat <<POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],
      "Resource": "arn:aws:logs:${REGION}:*:*"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject","s3:ListBucket"],
      "Resource": ["arn:aws:s3:::${S3_BUCKET}","arn:aws:s3:::${S3_BUCKET}/*"]
    },
    {
      "Effect": "Allow",
      "Action": ["rekognition:DetectFaces"],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": ["dynamodb:PutItem","dynamodb:Query"],
      "Resource": "arn:aws:dynamodb:${REGION}:${ACCOUNT_ID}:table/${DYNAMODB_TABLE}"
    },
    {
      "Effect": "Allow",
      "Action": ["appsync:GraphQL"],
      "Resource": "*"
    }
  ]
}
POLICY
)
        aws iam put-role-policy \
            --role-name "$LAMBDA_ROLE_NAME" \
            --policy-name "${PROJECT}-lambda-policy" \
            --policy-document "$POLICY_DOC"
        echo "  ポリシーアタッチ完了"
        echo "  IAM の反映を待機中 (10秒)..."
        sleep 10
    fi
fi

# ── 4. AppSync API ──────────────────────────────────
echo "[4/7] AppSync..."
APPSYNC_ID=""
EXISTING_API=$(aws appsync list-graphql-apis --region "$REGION" \
    --query "graphqlApis[?name=='${APPSYNC_NAME}'].apiId" --output text 2>/dev/null || true)

if [ -n "$EXISTING_API" ] && [ "$EXISTING_API" != "None" ]; then
    APPSYNC_ID="$EXISTING_API"
    echo "  既に存在: ${APPSYNC_ID}"
else
    APPSYNC_ID=$(aws appsync create-graphql-api \
        --name "$APPSYNC_NAME" \
        --authentication-type API_KEY \
        --region "$REGION" \
        --query 'graphqlApi.apiId' --output text)
    echo "  作成完了: ${APPSYNC_ID}"
fi

# API キー作成 (有効期限: 7日)
EXPIRY=$(date -v+7d +%s 2>/dev/null || date -d "+7 days" +%s)
EXISTING_KEYS=$(aws appsync list-api-keys --api-id "$APPSYNC_ID" --region "$REGION" \
    --query 'apiKeys[0].id' --output text 2>/dev/null || true)

if [ -n "$EXISTING_KEYS" ] && [ "$EXISTING_KEYS" != "None" ]; then
    APPSYNC_API_KEY=$(aws appsync list-api-keys --api-id "$APPSYNC_ID" --region "$REGION" \
        --query 'apiKeys[0].id' --output text)
    echo "  API キー既に存在"
else
    APPSYNC_API_KEY=$(aws appsync create-api-key \
        --api-id "$APPSYNC_ID" \
        --expires "$EXPIRY" \
        --region "$REGION" \
        --query 'apiKey.id' --output text)
    echo "  API キー作成完了"
fi

# スキーマ定義
echo "  スキーマをアップロード中..."
SCHEMA_FILE="${PROJECT_ROOT}/infra/appsync/schema.graphql"
aws appsync start-schema-creation \
    --api-id "$APPSYNC_ID" \
    --definition "fileb://${SCHEMA_FILE}" \
    --region "$REGION" --output text > /dev/null

echo "  スキーマ反映を待機中..."
for i in $(seq 1 30); do
    STATUS=$(aws appsync get-schema-creation-status \
        --api-id "$APPSYNC_ID" --region "$REGION" \
        --query 'status' --output text 2>/dev/null || echo "PROCESSING")
    if [ "$STATUS" = "SUCCESS" ] || [ "$STATUS" = "ACTIVE" ]; then
        echo "  スキーマ反映完了"
        break
    elif [ "$STATUS" = "FAILED" ]; then
        echo "  [ERROR] スキーマ反映失敗"
        aws appsync get-schema-creation-status --api-id "$APPSYNC_ID" --region "$REGION"
        exit 1
    fi
    sleep 2
done

# NONE データソース + リゾルバ作成 (publishClapEvent は Local Resolver)
DS_NAME="NoneDataSource"
EXISTING_DS=$(aws appsync list-data-sources --api-id "$APPSYNC_ID" --region "$REGION" \
    --query "dataSources[?name=='${DS_NAME}'].name" --output text 2>/dev/null || true)

if [ -z "$EXISTING_DS" ] || [ "$EXISTING_DS" = "None" ]; then
    aws appsync create-data-source \
        --api-id "$APPSYNC_ID" \
        --name "$DS_NAME" \
        --type NONE \
        --region "$REGION" --output text > /dev/null
    echo "  NONE データソース作成完了"
fi

# publishClapEvent リゾルバ (パススルー)
EXISTING_RESOLVER=$(aws appsync get-resolver \
    --api-id "$APPSYNC_ID" \
    --type-name "Mutation" \
    --field-name "publishClapEvent" \
    --region "$REGION" 2>/dev/null && echo "exists" || echo "")

if [ -z "$EXISTING_RESOLVER" ]; then
    aws appsync create-resolver \
        --api-id "$APPSYNC_ID" \
        --type-name "Mutation" \
        --field-name "publishClapEvent" \
        --data-source-name "$DS_NAME" \
        --request-mapping-template '{"version":"2017-02-28","payload":$util.toJson($context.arguments.input)}' \
        --response-mapping-template '$util.toJson($context.result)' \
        --region "$REGION" --output text > /dev/null
    echo "  publishClapEvent リゾルバ作成完了"
fi

APPSYNC_ENDPOINT=$(aws appsync get-graphql-api --api-id "$APPSYNC_ID" --region "$REGION" \
    --query 'graphqlApi.uris.GRAPHQL' --output text)

# ── 5. Lambda 関数 ──────────────────────────────────
echo "[5/7] Lambda..."

# デプロイパッケージ作成
LAMBDA_DIR="${PROJECT_ROOT}/lambda"
DEPLOY_DIR=$(mktemp -d)
cp "${LAMBDA_DIR}/handler.py" "$DEPLOY_DIR/"
(cd "$DEPLOY_DIR" && zip -q handler.zip handler.py)

EXISTING_LAMBDA=$(aws lambda get-function --function-name "$LAMBDA_NAME" --region "$REGION" 2>/dev/null && echo "exists" || echo "")

if [ -n "$EXISTING_LAMBDA" ]; then
    aws lambda update-function-code \
        --function-name "$LAMBDA_NAME" \
        --zip-file "fileb://${DEPLOY_DIR}/handler.zip" \
        --region "$REGION" --output text > /dev/null
    echo "  コード更新完了: ${LAMBDA_NAME}"
    aws lambda wait function-updated --function-name "$LAMBDA_NAME" --region "$REGION" 2>/dev/null || sleep 5
    aws lambda update-function-configuration \
        --function-name "$LAMBDA_NAME" \
        --environment "Variables={S3_BUCKET_NAME=${S3_BUCKET},DYNAMODB_TABLE=${DYNAMODB_TABLE},APPSYNC_ENDPOINT=${APPSYNC_ENDPOINT},APPSYNC_API_KEY=${APPSYNC_API_KEY},APP_REGION=${REGION},S3_COMPOSED_PREFIX=composed/}" \
        --region "$REGION" --output text > /dev/null
    echo "  環境変数更新完了"
else
    aws lambda create-function \
        --function-name "$LAMBDA_NAME" \
        --runtime python3.11 \
        --handler handler.lambda_handler \
        --role "$LAMBDA_ROLE_ARN" \
        --zip-file "fileb://${DEPLOY_DIR}/handler.zip" \
        --timeout 30 \
        --memory-size 256 \
        --environment "Variables={S3_BUCKET_NAME=${S3_BUCKET},DYNAMODB_TABLE=${DYNAMODB_TABLE},APPSYNC_ENDPOINT=${APPSYNC_ENDPOINT},APPSYNC_API_KEY=${APPSYNC_API_KEY},APP_REGION=${REGION},S3_COMPOSED_PREFIX=composed/}" \
        --region "$REGION" --output text > /dev/null
    echo "  作成完了: ${LAMBDA_NAME}"
fi

rm -rf "$DEPLOY_DIR"

# ── 6. S3 → Lambda イベント通知 ─────────────────────
echo "[6/7] S3 イベント通知..."

# Lambda に S3 からの呼び出し許可を付与
aws lambda add-permission \
    --function-name "$LAMBDA_NAME" \
    --statement-id "s3-trigger-${PROJECT}" \
    --action lambda:InvokeFunction \
    --principal s3.amazonaws.com \
    --source-arn "arn:aws:s3:::${S3_BUCKET}" \
    --region "$REGION" 2>/dev/null || true

LAMBDA_ARN=$(aws lambda get-function --function-name "$LAMBDA_NAME" --region "$REGION" \
    --query 'Configuration.FunctionArn' --output text)

NOTIFICATION_CONFIG=$(cat <<NOTIF
{
  "LambdaFunctionConfigurations": [
    {
      "LambdaFunctionArn": "${LAMBDA_ARN}",
      "Events": ["s3:ObjectCreated:*"],
      "Filter": {
        "Key": {
          "FilterRules": [
            {"Name": "prefix", "Value": "raw/"}
          ]
        }
      }
    }
  ]
}
NOTIF
)

aws s3api put-bucket-notification-configuration \
    --bucket "$S3_BUCKET" \
    --notification-configuration "$NOTIFICATION_CONFIG" \
    --region "$REGION"
echo "  設定完了: raw/ → ${LAMBDA_NAME}"

# ── 7. 結果出力 ────────────────────────────────────
echo ""
echo "[7/7] セットアップ完了!"
echo ""
echo "=========================================="
echo " 以下を .env に設定してください"
echo "=========================================="
echo ""
echo "# AWS"
echo "S3_BUCKET_NAME=${S3_BUCKET}"
echo "DYNAMODB_TABLE=${DYNAMODB_TABLE}"
echo "APPSYNC_ENDPOINT=${APPSYNC_ENDPOINT}"
echo "APPSYNC_API_KEY=${APPSYNC_API_KEY}"
echo ""
echo "# dashboard/.env.local"
echo "NEXT_PUBLIC_APPSYNC_ENDPOINT=${APPSYNC_ENDPOINT}"
echo "NEXT_PUBLIC_APPSYNC_API_KEY=${APPSYNC_API_KEY}"
echo "NEXT_PUBLIC_AWS_REGION=${REGION}"
echo ""
echo "=========================================="
