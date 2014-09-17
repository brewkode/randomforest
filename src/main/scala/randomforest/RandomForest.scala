package randomforest

case class RandomForestModel(trees: Iterable[Node]) {
  def predict(featureVector: Array[Double]) = {
    val predictions: Iterable[Prediction] = trees.map(tree => tree.predict(featureVector))
      predictions.map(_.predictedLabel)
      .groupBy(identity)
      .maxBy(_._2.size)
      ._1
  }
  def numTrees = trees.size
}

class RandomForest(numTrees: Int, maxDepth: Int, featureSampleSize:Int, sampleSizePerTree:Int) {
  def train(labelledData:IndexedSeq[LabelledData]) = {
    RandomForestModel(
      (1 to numTrees).map(i => new DecisionTreeTrainer(maxDepth, featureSampleSize, sampleSizePerTree).train(labelledData))
    )
  }
}
