package org.ucf.ml
package parser

import java.io.IOException
import java.nio.file.{Files, Paths}

abstract class Granularity extends utils.context {
  protected def readSourceCode(input:String) = {
    val reg:String = try {
      if(Files.exists(Paths.get(input))){
        logger.debug(f"This input File Path: ${input}")
        new String(Files.readAllBytes(Paths.get(input)))
      } else {
        logger.debug("The input is a string source code")
        input
      }
    } catch {
      case e:IOException => {
        e.printStackTrace()
        EMPTY_STRING
      }
    }
    reg.replaceAll("(?:/\\*(?:[^*]|(?:\\*+[^*/]))*\\*+/)|(?://.*)","") //remove comments
       .replaceAll("@.+", "") //remove annotations
  }

  def getSourceCode(input:String=null):String
}

case class Granularity_Class(sourcePath:String) extends Granularity {
  override def getSourceCode(input: String=sourcePath): String = readSourceCode(input)
}

case class Granularity_Method(sourcePath:String) extends Granularity {
  override def getSourceCode(input: String=sourcePath): String =
    raw"public class DummyClass { ${readSourceCode(input)} }"
}


object Granularity {
  def apply(sourcePath:String, granularity: String): Granularity = granularity match {
    case "class" => Granularity_Class(sourcePath)
    case "method" => Granularity_Method(sourcePath)
    case _ => Granularity_Class(sourcePath)
  }
}