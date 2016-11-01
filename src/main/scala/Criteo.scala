import scala.io.Source
import scala.util.hashing.MurmurHash3
import breeze.linalg._
import breeze.numerics._

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
  * Created by yyao39 on 11/1/16.
  */
object Criteo {

  // parameters
  val train : String = "train.txt"
  val test : String = "test.txt"
  val logbatch : Int = 100000
  val D : Int = 16777216
  val lambda1 : Double = 0.0
  val lambda2 : Double = 0.0
  val alpha : Double = 0.05
  val adapt : Double = 1.0
  val fudge : Double = 0.5

  // initialize our model
  // weights = [0.] * D
  val w = DenseVector.rand[Double](D)

  // sum of historical gradients = [fudge] * D
  val g = DenseVector.ones[Double](D) :* fudge

  def main(args : Array[String]): Unit = {

    // start training a logistic regression model using on pass sgd
    var loss: Double = 0.0
    var lossb: Double = 0.0
    val content = Source.fromFile("sample0.features").getLines.map(_.split(","))
    //val content = Source.fromFile("train.csv").getLines.map(_.split(","))
    val header = content.next()
    //val header = "Label,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,c1,c2,c3,c4,c5,c6,c7" +
    //  ",c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26"
    val content_map = content.map(header.zip(_).toMap)
    var num_line = 0

    // main training procedure
    while (content_map.hasNext) {
      num_line += 1
      var row = content_map.next()
      //println(row)
      var y: Double = 0.0
      if (row("Label") == "1")
        y = 1.0
      //println(y)
      row -= "Label"
      //println(row)

      // step 1, get the hashed features
      val x = get_x(row, D)

      // step 2, get prediction
      val p = get_p(x, w)

      // for progress validation, useless for learning our model
      val lossx: Double = logloss(p, y)
      loss += lossx
      lossb += lossx

      if (num_line % logbatch == 0) {
        println("reach: ", num_line, "\t total loss: ", loss / num_line)
        lossb = 0.0
      }

      // step 3, update model with answer
      update_w(w, g, x, p, y)
    }
  }

  def get_x(row: Map[String, String], D: Int): mutable.HashMap[Int, Int] = {
      var fullind = ListBuffer[Int]()
      for ((k, v) <- row) fullind += MurmurHash3.stringHash(k + "=" + v) % D
      val x = new mutable.HashMap[Int, Int]

      for (index <- fullind) {
        if (x.contains(index)) x(index) += 1
        else x.+=((index, 1))
      }

      x
  }

  def get_p(x: mutable.HashMap[Int, Int], w: DenseVector[Double]): Double = {
      var wTx = 0.0
      for ((i, xi) <- x) wTx += w(i) * xi
      1 / (1 + Math.exp(-Math.max(Math.min(wTx, 50.0), -50.0)))
  }


  def logloss(p : Double, y : Double) : Double = {
    val p_bounded = Math.max(Math.min(p, 1.0 - 10e-17), 10e-17)
    if (y == 1.0)
      -1 * Math.log(p_bounded)
    else
      -1 * Math.log(1.0 - p_bounded)
  }

  def update_w(w : DenseVector[Double], g : DenseVector[Double],
               x : mutable.HashMap[Int, Int], p : Double, y : Double) : Unit = {
    val delreg = 0

    for((i,xi) <- x) {
      //delreg = (lambda1 * ((-1.) if w(i) < 0. else 1.) +lambda2 * wi])
      val delta = (p - y) * xi + delreg
      if(adapt > 0) g(i) += delta * delta
      w(i) -= delta * alpha / Math.pow(sqrt(g(i)), adapt)
    }
  }
}
