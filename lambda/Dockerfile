# AWS Lambda Python 3.11 ランタイムと同一のベースイメージを使用
# 本番デプロイ時と同じ実行環境を再現する
FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ソースコードは docker-compose の volume mount で提供
# （本番デプロイ時は COPY . ${LAMBDA_TASK_ROOT} に変更すること）

CMD ["handler.lambda_handler"]
