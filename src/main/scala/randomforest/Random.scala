package randomforest

import scala.util.Random

trait Random {
  lazy val choice = new RandomChoice
}


class RandomChoice {
  def nextDouble() = Random.nextDouble()
  def nextInt() = Random.nextInt()
  def nextInt(range: Int) = Random.nextInt(range)
  def shuffle[T](iterable: Iterable[T]) = Random.shuffle(iterable)
}
