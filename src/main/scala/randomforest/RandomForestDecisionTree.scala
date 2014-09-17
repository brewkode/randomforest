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
case class LabelledData(label: String, featureVector: Array[Double]){
  def tsv() = label+"\t"+featureVector.mkString("\t")
}
case class Split(left:IndexedSeq[LabelledData],right:IndexedSeq[LabelledData],informationGain:Double,threshold:Double,featureIndex:Int)
case class Prediction(probabilities: Map[String, Double]) {
  def probabilityFor(label: String) = probabilities(label)
  def predictedLabel = probabilities.maxBy(_._2)._1
}

trait Node {
  def predict(featureVector:Array[Double]):Prediction
}
case class LeafNode(labelProbablities:Map[String,Double]) extends Node {
  def predict(featureVector:Array[Double]) = Prediction(labelProbablities)
}
case class SplitNode(left:Node,right:Node, featureIndex: Int, threshold: Double) extends Node {
  def predict(featureVector:Array[Double]) = {
    if(featureVector(featureIndex) < threshold) left.predict(featureVector) else right.predict(featureVector)
  }
}

class DecisionTreeTrainer(maxDepth: Int, numberOfFeaturesToSample:Int, sizeOfSampleBag:Int)(implicit bootstrap: Bootstrap = new Bootstrap()) {
  def train(labelledData:IndexedSeq[LabelledData]) = {
    val featureDimensions = labelledData.head.featureVector.size
    buildTree(bootstrap.sampleWithReplacement(math.min(labelledData.size,sizeOfSampleBag))(labelledData),math.min(maxDepth,featureDimensions))
  }

  def buildTree(sampledData:IndexedSeq[LabelledData],depth:Int):Node = {
    val sampleFeatureIndices = bootstrap.sampleFeatures(math.min(numberOfFeaturesToSample,sampledData.head.featureVector.size))(sampledData.head.featureVector.size)
    val entropyOfDataset = FrequencyHistogram.entropyInLabels(sampledData)

    if(depth == 0) return LeafNode(FrequencyHistogram.labelPriors(sampledData))

    val split = sampleFeatureIndices.map(featureIdx => {
      val threshold = bootstrap.guessThreshold()
      val (left, right) = sampledData.partition(_.featureVector(featureIdx) < threshold)

      val informationGain = entropyOfDataset - (FrequencyHistogram.entropyInLabels(left) * left.size + FrequencyHistogram.entropyInLabels(right) * right.size) / sampledData.size
      Split(left, right, informationGain, threshold, featureIdx)
    }).maxBy(_.informationGain)

    if(math.abs(split.informationGain) < 0.001) return LeafNode(FrequencyHistogram.labelPriors(sampledData))
    SplitNode(buildTree(split.left,depth -1),buildTree(split.right,depth -1), split.featureIndex, split.threshold)
  }
}

class Bootstrap extends Random {
  def sampleWithReplacement(sampleSize:Int)(data:IndexedSeq[LabelledData]): IndexedSeq[LabelledData] = {
    1.to(sampleSize).map(e => choice.nextInt(data.size-1)).map(data)
  }

  def sampleFeatures(sampleSize:Int)(noOfFeatures:Int) = {
    choice.shuffle(0.to(noOfFeatures-1).toList).take(sampleSize)
  }

  def guessThreshold() = choice.nextDouble
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
