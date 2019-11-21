package org.ucf.ml

/**
  * @author 
  */
object ScalaApp {
  val log = new org.ucf.ml.utils.Log(this.getClass.getName)
  def printHello() =  log.info("Hello World from Scala")
  def main(args: Array[String]): Unit = {
    printHello()
  }
}