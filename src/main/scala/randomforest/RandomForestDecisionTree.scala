package randomforest


/*
  0. Settings - max depth
  1. Input data set as Iterable[LabelledData]
  2. Sample input
  3. Sample features
  4. Identify feature that min loss. i.e., iG or  gini coeff
  5. Goto step 3 until max depth
 */
case class LabelledData(label: String, featureVector: Array[Int])
class RandomForestDecisionTree(maxDepth: Int) {

}
