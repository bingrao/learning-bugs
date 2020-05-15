package org.ucf.ml
package parser

import java.io.IOException
import java.nio.file.{Files, Paths}

abstract class Granularity extends utils.Common {
  protected def readSourceCode(input:String) = {
    val reg:String = try {
      if(Files.exists(Paths.get(input))){
        new String(Files.readAllBytes(Paths.get(input)))
      } else {
        input
      }
    } catch {
      case e:IOException => {
        e.printStackTrace()
        EmptyString
      }
    }
    // Remove all comments and anotations in the source code
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
  def apply(sourcePath:String, granularity: Value): Granularity = granularity match {
    case CLASS => Granularity_Class(sourcePath)
    case METHOD => Granularity_Method(sourcePath)
    case _ => Granularity_Class(sourcePath)
  }
}