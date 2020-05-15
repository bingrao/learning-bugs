package org.ucf.ml.utils

import java.io.{File, FileWriter, PrintWriter}

trait Common {

  val logger = new Logging(this.getClass.getName)

  def write(path:String, context:String) = {
    val printWriter = new PrintWriter(new FileWriter(path))
    printWriter.print(context)
    printWriter.close()
  }

  def getListOfFiles(dir: String):List[File] = {
    val d = new File(dir)
    if (d.exists && d.isDirectory) {
      d.listFiles.filter(_.isFile).toList
    } else {
      List[File]()
    }
  }
}
