package org.ucf.ml.plaintext

import java.nio.file.{Files, Paths}
import java.io.{File, IOException}

import scala.collection.mutable.ListBuffer

object CodeTranslator {
  def translate(lexedCodeFile:String, out:String, mapsDir:String) = {
    val translatedCodeLines = new ListBuffer[String]()
    val lines = try {
      Files.readAllLines(Paths.get(lexedCodeFile))
    } catch {
      case e:IOException => e.printStackTrace()
    }
    val lexedCodeLines = lines.asInstanceOf[List[String]]
    for(i <- lexedCodeLines.indices) {
      val codePrinter = new CodePrinter(mapsDir+File.separator+i)
      val code = codePrinter.printCode(lexedCodeLines(i))
      translatedCodeLines.+=(code)
    }

//    try {
//      Files.write(Paths.get(out), translatedCodeLines.toArray[Byte])
//    } catch {
//      case e:IOException => e.printStackTrace()
//    }
  }
}
