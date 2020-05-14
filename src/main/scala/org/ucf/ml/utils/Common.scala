package org.ucf.ml.utils

import java.io.{FileWriter, PrintWriter}
import java.nio.file.{Files, Paths}
import java.util.stream.Collectors
import scala.collection.mutable
import scala.collection.JavaConversions._

trait Common {
  final val EMPTY_STRING = ""
  val logger = new Logging(this.getClass.getName)

  def write(path:String, context:String) = {
    val printWriter = new PrintWriter(new FileWriter(path))
    printWriter.print(context)
//    printWriter.append(context)
    printWriter.close()
  }

  def readIdioms(filePath:String) = {
    var idioms = new mutable.HashSet[String]()
    try{
      val stream = Files.lines(Paths.get(filePath))
      idioms.++=(stream.collect(Collectors.toSet[String]()))
    } catch {
      case e:Exception => e.printStackTrace()
    }
    idioms
  }
}
