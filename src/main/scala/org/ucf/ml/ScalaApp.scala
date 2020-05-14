package org.ucf.ml

/**
  * @author 
  */
object ScalaApp {
  def main(args: Array[String]): Unit = {
    val worker = new parallel.Master()
    worker.run()
  }
}