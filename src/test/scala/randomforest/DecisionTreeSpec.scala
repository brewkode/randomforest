package randomforest

import org.scalatest.FlatSpec
import org.scalatest.matchers.ShouldMatchers


class DecisionTreeSpec extends FlatSpec with ShouldMatchers {
  "FrequencyHistogram" should "find frequency histogram of feature label combination" in {
    val histogram = FrequencyHistogram.histogram(List(
      LabelledData("label1", Array[Double](0, 0, 1, 1)),
      LabelledData("label1", Array[Double](0, 0, 1, 1)),
      LabelledData("label2", Array[Double](0, 11, 0, 0)),
      LabelledData("label2", Array[Double](0, 2, 121, 0))
    ))

    histogram("label2") should be(2)
    histogram("label1") should be(2)
  }

  it should "find entropy in the labels in the set of labelled data" in {
    val entropy= FrequencyHistogram.entropyInLabels(List(
      LabelledData("label1", Array[Double](0, 0, 1, 1)),
      LabelledData("label1", Array[Double](0, 0, 1, 1)),
      LabelledData("label2", Array[Double](0, 11, 0, 0))
    ))

    entropy should be(DoubleTolerance(0.918,0.001))
  }

  "Decision" should "make decisions" in {
    val trainer = new DecisionTreeTrainer(1,10,4)
    val decisionTree = trainer.train(Array(
      LabelledData("label1", Array[Double](0.3, 0.1, 0.1, 0.1)),
      LabelledData("label1", Array[Double](0.1, 0.1, 0.1, 0.1)),
      LabelledData("label2", Array[Double](0.1, 0.5, 0.2, 0.1)),
      LabelledData("label2", Array[Double](0.3, 0.4, 0.7, 0.1))
    ))

    print(decisionTree.predict(Array[Double](1, 2, 0, 1)))
  }
}
