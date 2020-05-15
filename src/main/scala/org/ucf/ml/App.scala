package org.ucf.ml

/**
  * @author 
  */
object App {
  def main(args: Array[String]): Unit = {
    val worker = new parallel.Master()
    worker.run()
  }
}