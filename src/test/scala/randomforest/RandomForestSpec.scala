package randomforest

import org.scalatest.FlatSpec
import org.scalatest.matchers.ShouldMatchers._
import java.io._

class RandomForestSpec extends FlatSpec {
  val labelledData = Array(
    LabelledData("label1", Array[Double](0, 0, 1, 1)),
    LabelledData("label1", Array[Double](0, 0, 1, 1)),
    LabelledData("label2", Array[Double](0, 11, 4, 0)),
    LabelledData("label2", Array[Double](1, 11, 12, 1))
  )

  "Random Forest" should "build the model with specified number of trees" in {
    val forest = new RandomForest(2, 10, 4, 4)
    val randomForestModel = forest.train(labelledData)
    randomForestModel.predict(labelledData(1).featureVector) should be("label1")
    randomForestModel.predict(labelledData(3).featureVector) should be("label2")
  }

  it should "build model for a given dataset and make predictions on train set" in {
    val dataset = readSerFile(this.getClass.getResource("/randomforest_serialized_input.ser").getPath)
    val forest = new RandomForest(3, 17, 137, 1000)
    val randomForestModel = forest.train(dataset)
    val results = dataset.map(d => randomForestModel.predict(d.featureVector))
    println(results.zip(dataset).count{case (r, d) => r == d.label})
    println(results.zip(dataset).count{case (r, d) => r != d.label})
  }

  def readSerFile(file: String) = {
    val fis = io.Source.fromFile(this.getClass.getResource("/randomforest_serialized_input.ser").getPath).getLines
    fis.map{l =>
      val parts = l.split("\t")
      val (label, features) = parts.splitAt(1)
      LabelledData(label.head, features.map(_.toDouble))
    }.toArray
  }

  ignore should "process text file" in {
    val lines = io.Source.fromFile("/media/data/datasets/analysis/classifier/2013-10-31_titles_30k.tsv").getLines().map(_.toLowerCase)
    val wordIndex = lines
      .flatMap(_.split("\t")(0).split(" "))
      .toList
      .groupBy(identity)
      .filter(_._2.size > 25)
      .map(_._1)
      .zipWithIndex
      .map{case (ele, index) => (ele.toString, index)}
      .toMap

    val numFeatures = wordIndex.size

    def featureVector(numFeatures: Int, words:Array[String]) = {
      val indices = words.map(w => wordIndex.getOrElse(w, -1)).filter(_ != -1).toSet
      (0 to numFeatures-1).map(i => if(indices contains i) 1.0 else 0.0)
    }

    val records = io.Source.fromFile("/media/data/datasets/analysis/classifier/2013-10-31_titles_30k.tsv").getLines().map(_.toLowerCase)
    val features = records
      .map(_.split("\t"))
      .filter(_.length==2)
      .filter(t => !t(1).isEmpty && !t(1).equals("null"))
      .toList
      .map{ t =>
        val label = t(1)
        val words = t(0).split(" ")
        LabelledData(label, featureVector(numFeatures, words).toArray)
      }
      .filter(_.featureVector.exists(_ == 1.0))

    val oos = new FileWriter("/media/data/datasets/analysis/classifier/randomforest_serialized_input.ser")
    features.foreach(f => oos.write(f.tsv()+"\n"))
    oos.close()
  }
}
