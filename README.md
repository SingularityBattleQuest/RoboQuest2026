# RoboQuest2026

## 学習ノートブック

- [クイックスタート編](https://colab.research.google.com/github/SingularityBattleQuest/RoboQuest2026/blob/main/notebooks/quickstart_ja.ipynb)：上から実行して動かす。主要なパラメーターをスライダーで調整できます。
- [解説・実験編](https://colab.research.google.com/github/SingularityBattleQuest/RoboQuest2026/blob/main/notebooks/guide_and_experiments_ja.ipynb)：同じ学習フローを、観測・行動・報酬・PPOの解説と実験例付きで学びます。

Google Colabで実行してください。両教材とも「歩行 → 逃げ」の階層型PPOを使い、モデル・正規化データ・学習設定の保存形式は共通です。実験ごとに保存先を分けて比較できます。

本番は四脚Go2の鬼1体対逃げ1体です。両教材で鬼の追跡動画、本番用歩行・逃げ学習、提出ZIPの生成と検査、公式20試合の評価・試合動画まで進めます。確定ルール・観測・提出・本番実行方法は [公式仕様](competition/RULES.md) を参照してください。旧モデルは互換性対象外です。

### ローカルでの歩行学習とColabへの持ち込み

環境の修正内容、学習の再現手順、評価結果は [ローカル歩行学習](docs/walk_training/README.md) にまとめています。入門の同梱モデルは停止・前進の確認用です。本番では別の47次元入力モデルを学習し、前後進・左右旋回・横移動・停止の資格試験を通過してから逃げ学習へ進みます。

初期設定の `retrain_walk=True` は未学習から学習します。`retrain_walk=False` に変更すると、同梱した停止・前進用の検証済みモデルを読み込みます。ZIPのアップロードは不要です。

本番用の鬼は学習済み [`competition/oni/walk.onnx`](competition/oni/walk.onnx) を両Colabと本番で共通に読み込みます。SHA-256を照合し、固定の追跡速度指令から12関節を動かします。鬼の再学習は不要です。学習手順と追跡試験結果は [`competition/oni/manifest.json`](competition/oni/manifest.json) に保存しています。
