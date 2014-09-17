package randomforest

import org.scalatest.FlatSpec
import org.scalatest.matchers.ShouldMatchers
import org.mockito.Mockito._
import scala.collection.mutable


class DecisionTreeSpec extends FlatSpec with ShouldMatchers {
  val mockChoice = mock(classOf[RandomChoice])
  trait MockRandom extends Random{
    override lazy val choice = mockChoice
  }
  val labelledData = Array(
    LabelledData("label1", Array[Double](0, 0, 1, 1)),
    LabelledData("label1", Array[Double](0, 0, 1, 1)),
    LabelledData("label2", Array[Double](0, 11, 0, 0))
  )

  "FrequencyHistogram" should "find frequency histogram of feature label combination" in {
    val histogram = FrequencyHistogram.histogram(labelledData)
    histogram("label2") should be(1)
    histogram("label1") should be(2)
  }

  it should "find entropy in the labels in the set of labelled data" in {
    val entropy= FrequencyHistogram.entropyInLabels(labelledData)
    entropy should be(DoubleTolerance(0.918,0.001))
  }

  "Bootstrap" should "sample a list with replacement" in {
    when(mockChoice.nextInt(2)).thenReturn(1).thenReturn(1).thenReturn(2)
    val bootstrap = new Bootstrap() with MockRandom
    bootstrap.sampleWithReplacement(3)(labelledData) should be(List(labelledData(1),labelledData(1),labelledData(2)))
  }

  it should "sample feature indices" in {
    when(mockChoice.shuffle(0 to 3)).thenReturn(List(2,3,1,0))
    val bootstrap = new Bootstrap() with MockRandom
    bootstrap.sampleFeatures(3)(4) should be(List(2,3,1))
  }

  "DecisionTree" should "make decisions" in {
    val data = Array(
      LabelledData("label1", Array[Double](0.3, 0.1, 0.1, 0.1)),
      LabelledData("label1", Array[Double](0.1, 0.1, 0.1, 0.1)),
      LabelledData("label2", Array[Double](0.1, 0.5, 0.2, 0.1)),
      LabelledData("label2", Array[Double](0.3, 0.4, 0.7, 0.1))
    )
    implicit val bootstrap = mock(classOf[Bootstrap])
    when(bootstrap.sampleFeatures(4)(4)).thenReturn(List(2,3,1))
    when(bootstrap.sampleWithReplacement(4)(data)).thenReturn(Array(data(2),data(0),data(1)))
    when(bootstrap.guessThreshold()).thenReturn(0.2).thenReturn(0.3).thenReturn(0.5)
    val trainer = new DecisionTreeTrainer(1,10,4)
    val decisionTree = trainer.train(data)

    decisionTree.predict(data(0).featureVector).predictedLabel should be("label1")
    decisionTree.predict(data(1).featureVector).predictedLabel should be("label1")
    decisionTree.predict(data(2).featureVector).predictedLabel should be("label2")
    decisionTree.predict(data(3).featureVector).predictedLabel should be("label2")
  }
}
