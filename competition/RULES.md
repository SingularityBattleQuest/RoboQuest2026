# RQ2026 確定ルールと実行仕様

仕様バージョン: `rq2026-1v1-v1`。旧モデルは互換性対象外です。

## 試合・順位

- Go2の鬼1体対Go2の逃げ1体。各チームが個別に対戦します。
- 5m×5mの平坦なフィールド、外周に壁、障害物なし。逃げは中央・向き0rad、鬼は中央から2m、中央を向いて開始します。
- 東・北・西・南の4条件×seed 100〜104の5条件、計20試合。最大60秒、シミュレーション時間で計測します。
- 2体の衝突形状が接触した時点で捕獲。逃げの胴体高さ0.15m未満が0.5秒継続、または胴体中心のx/yが±2.5mを超えると終了します。鬼の転倒で計時は止めません。
- 順位は平均生存時間のみ。距離ボーナスなし。同点は鬼の目標速度0.4m/s、次に0.5m/sで各20試合。それでも同点なら同順位・同率優勝。全試合で同一の提出物を使います。
- 通常の鬼の目標速度は0.3m/s。固定の追跡フィードバックと逆運動学によるトロットを関節制御で実行します。位置の直接移動・瞬間的な向き変更は行いません。実際の速度は接触・旋回・物理状態に依存します。
- 運営基盤の障害は同条件で再試合。モデル異常・推論制限超過は発生時点で終了し、理由を記録します。
- 公式環境の公開から締切まで最低7日。必要に応じ暫定10月20日の締切を延長します。期限内の最後の有効提出物を採用します。

## 提出

Colabの提出セルで `submission.zip` を生成します。Python・pickle・外部ファイルは本番で実行しません。正規化はONNXに含めます。

| 形式 | ファイル | float32 入力→出力 |
|---|---|---|
| 階層型 | walk.onnx / flee.onnx / manifest.json | 47→12 / 19→3 |
| 直接制御型 | direct.onnx / manifest.json | 66→12 |

全モデルは `[1,N]` の観測を受け、`[1,M]` の行動を返します。動的batchも1を受け付ければ可能です。行動はすべて[-1,1]。関節目標は立位姿勢＋行動×0.6radで、関節可動域で制限します。位置ゲイン60、関節減衰2、各脚のトルク上限23.7/23.7/45.43Nm。物理刻み0.002秒、低レベル制御50Hz、高レベル5Hz。

高レベルの出力3値はそれぞれ前後×0.5m/s、左右×0.3m/s、旋回×1rad/sへ変換します。

manifest.jsonは version、team、mode、sources、各モデルのsha256を記載します。利用権限のある既存コード・モデルは使用可能で、出典と利用範囲をsourcesに記載します。PPO以外のアルゴリズム、見本学習、独自ネットワーク、報酬変更、学習環境の改変も可能です。本番では提供観測・固定物理条件のみを使います。

制限はZIP/展開後各100MiB、CPUのみ、外部通信なし、1スレッド、メモリ2GiB、初回準備を除く各推論100ms。ONNXの標準演算の許可リストと形状・有限値・ファイルの検査はsubmission.pyが定義します。

## 観測順序

高レベル19次元:

| index | 意味・単位 |
|---|---|
| 0,1 | 自機基準の鬼相対位置x,y（m） |
| 2 | 鬼の水平距離（m） |
| 3,4 | 自機の世界位置x,y（m） |
| 5,6 | 自機の向きcos,sin |
| 7,8,9 | 自機基準の速度x,y,z（m/s） |
| 10,11,12,13 | 東・北・西・南の壁までの距離（m） |
| 14 | 残り時間÷60秒 |
| 15,16 | 自機基準の鬼との相対速度x,y（m/s） |
| 17 | 鬼の胴体高さ（m） |
| 18 | 鬼の直立度（回転行列zz） |

低レベル47次元: 速度指令3、角速度3（rad/s、MuJoCo自由関節の回転速度）、自機基準の重力方向3、立位との差の関節角12（rad）、関節速度12（rad/s）、直前の関節行動12、sin/cos(2π×1.5×試合時間)2。関節順序はFR/FL/RR/RLのhip/thigh/calfです。公開の位相入力で学習した歩行周期を進めます。入門の前進45次元モデルと区別してください。

直接制御66次元: 指令を0にした低レベル47次元の後に高レベル19次元。低レベルの直前行動は直接制御の直前行動です。

## 再現・実行

学習と採点は同じ `competition.arena.Arena` を使います。XMLとメッシュの共通hash、ルールhash、提出hash、依存バージョン、各試合の終了理由を結果JSONへ保存します。

```sh
python -m competition.run_match --submission submission.zip --output result.json
python -m competition.run_match --teams-dir submissions --output ranking.json
```

本番は固定のLinux/amd64コンテナで、提出物を読み取り専用、ネットワーク無効、メモリ2GiB/1CPUで実行します。開催者は公開したコミットのイメージdigestを固定します。

```sh
docker build --platform linux/amd64 -f competition/Dockerfile -t rq2026-official .
mkdir -p results
chmod 777 results
docker run --rm --platform linux/amd64 --network none --memory 2g --cpus 1 \
  --read-only --cap-drop ALL --security-opt no-new-privileges \
  --tmpfs /tmp:rw,nosuid,nodev,size=128m \
  -v "$PWD/submissions:/submissions:ro" -v "$PWD/results:/results:rw" \
  rq2026-official --teams-dir /submissions --output /results/ranking.json
```

実行環境のビルド・評価の検証結果はVALIDATION.mdに記録します。提出物の準備に失敗した場合は採点開始前の検査失敗として修正・再提出の対象です。
