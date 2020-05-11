package org.ucf.ml
package utils

import java.io.{FileWriter, PrintWriter}

trait context {
  final val EMPTY_STRING = ""
  val logger = new Logging(this.getClass.getName)

  def write(path:String, context:String) = {
    val printWriter = new PrintWriter(new FileWriter(path))
    printWriter.print(context)
    printWriter.close()
  }
}
