# auto-design

## summary
- 配管などの設計を機械学習で自動生成することを検討する

## auto-design-using-Qlearning.ipynb
- Q-learningを使用した配管自動生成
- [todo]
  - 報酬系の変更
  - DQNでの学習方法に変更することを検討する
- [ref]
  - https://github.com/YutaroOgawa/Deep-Reinforcement-Learning-Book
  - program > 2_6_Qlearning.ipynb

  - 修正内容
    - コリジョンを線から、ブロックに変更
    - ゲームサイジングを自由に変更
    - コリジョン番号指定で、制約条件を自動生成
    - ゲームの結果を確認できるアニメーション追加
    - アニメーションのgif保存追加
