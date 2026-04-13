#!/bin/bash
# LocalStack 起動時に自動実行される初期化スクリプト
# S3バケットの作成など、ローカル開発に必要なリソースを事前構築する
#
# [Windows ユーザーへ]
# .gitattributes により改行コードはLFに固定されていますが、
# 万が一CRLFになった場合は以下で修正してください:
#   sed -i 's/\r//' infra/localstack-init/01_setup.sh
# または Git for Windows の設定:
#   git config --global core.autocrlf input

set -e

echo "[LocalStack Init] S3バケットを作成中..."
awslocal s3 mb s3://first-penguin-bucket
awslocal s3api put-object --bucket first-penguin-bucket --key raw/
awslocal s3api put-object --bucket first-penguin-bucket --key composed/

echo "[LocalStack Init] 完了"
