package randomforest

import scala.util.Random
import scala.collection.immutable


/*
  0. Settings - max depth
  1. Input data set as Iterable[LabelledData]
  2. Sample input
  3. Sample features
  4. Identify feature that min loss. i.e., iG or  gini coeff
  5. Goto step 3 until max depth
 */
case class LabelledData(label: String, featureVector: Array[Double])
case class Split(left:IndexedSeq[LabelledData],right:IndexedSeq[LabelledData],informationGain:Double,threshold:Double,featureIndex:Int)
trait Node {
  def predict(featureVector:Array[Double]):Map[String,Double]
}
case class LeafNode(labelProbablities:Map[String,Double]) extends Node {
  def predict(featureVector:Array[Double]) = labelProbablities
}
case class SplitNode(left:Node,right:Node,split:Split) extends Node {
  def predict(featureVector:Array[Double]) = {
    if(featureVector(split.featureIndex) < split.threshold) left.predict(featureVector) else right.predict(featureVector)
  }
}

class DecisionTreeTrainer(maxDepth: Int, numberOfFeaturesToSample:Int, sizeOfSampleBag:Int) {
  def train(labelledData:IndexedSeq[LabelledData]) = {
    val featureDimensions = labelledData.head.featureVector.size
    buildTree(Bootstrap.sampleWithReplacement(math.min(labelledData.size,sizeOfSampleBag))(labelledData),math.min(maxDepth,featureDimensions))
  }

  def buildTree(sampledData:IndexedSeq[LabelledData],depth:Int):Node = {
    val sampleFeatureIndices = Bootstrap.sampleFeatures(math.min(numberOfFeaturesToSample,sampledData.head.featureVector.size))(sampledData.head.featureVector.size)
    val entropyOfDataset = FrequencyHistogram.entropyInLabels(sampledData)

    if(depth == 0) return LeafNode(FrequencyHistogram.labelPriors(sampledData))

    val split = sampleFeatureIndices.map(featureIdx => {
      val threshold = Random.nextDouble()
      val (left, right) = sampledData.partition(_.featureVector(featureIdx) < threshold)

      val informationGain = entropyOfDataset - (FrequencyHistogram.entropyInLabels(left) * left.size + FrequencyHistogram.entropyInLabels(right) * right.size) / sampledData.size
      Split(left, right, informationGain, threshold, featureIdx)
    }).maxBy(_.informationGain)

    if(math.abs(split.informationGain) < 0.001) return LeafNode(FrequencyHistogram.labelPriors(sampledData))
    SplitNode(buildTree(split.left,depth -1),buildTree(split.right,depth -1),split)
  }
}

object Bootstrap {
  def sampleWithReplacement(sampleSize:Int)(data:IndexedSeq[LabelledData]) = {
    1.to(sampleSize).map(e => Random.nextInt(data.size-1)).map(data)
  }

  def sampleFeatures(sampleSize:Int)(noOfFeatures:Int) = {
   Random.shuffle(0.to(noOfFeatures-1).toList).take(sampleSize)
  }
}

object FrequencyHistogram {
  def histogram(labelledData:Iterable[LabelledData]):Map[String, Int] = {
    labelledData.map(_.label).groupBy(identity).map{case (k,vs) => k -> vs.size}
  }

  def labelPriors(labelledData:Iterable[LabelledData]) = {
    val totalSize = labelledData.size
    histogram(labelledData).map{case (k,v) => k -> v.toDouble/totalSize}
  }

  def entropyInLabels(labelledData:Iterable[LabelledData]) = {
    val priors = labelPriors(labelledData).map(_._2)
    - priors.map(d => d*math.log(d)/math.log(2)).sum
  }
}
