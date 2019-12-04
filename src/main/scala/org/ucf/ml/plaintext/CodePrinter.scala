package org.ucf.ml.plaintext
import scala.collection.mutable
import java.io.IOException
import java.nio.file.{Files, Paths}
class CodePrinter(mapFile: String) {
  private val identifiers = new mutable.HashMap[String,String]()
  private val stringLiteral = new mutable.HashMap[String,String]()
  private val characterLiteral = new mutable.HashMap[String,String]()
  private val integerLiteral = new mutable.HashMap[String,String]()
  private val floatingPointLiteral = new mutable.HashMap[String,String]()

  private val IDENT_PREFIX = "ID_"
  private val STRING_PREFIX = "STRING_"
  private val CHAR_PREFIX = "CHAR_"
  private val INT_PREFIX = "INT_"
  private val FLOAT_PREFIX = "FLOAT_"
  importMaps(mapFile)

  def printCode(lexedCode:String) =  {
    val sb = new mutable.StringBuilder()
    val tokens = lexedCode.split(" ")

    tokens.foreach(token => {
      if(token.startsWith(IDENT_PREFIX)) {
        sb.append(identifiers.get(token))
      } else if (token.startsWith(STRING_PREFIX)) {
        sb.append(stringLiteral.get(token))
      } else if (token.startsWith(CHAR_PREFIX)) {
        sb.append(characterLiteral.get(token))
      } else if (token.startsWith(INT_PREFIX)) {
        sb.append(integerLiteral.get(token))
      } else if (token.startsWith(FLOAT_PREFIX)) {
        sb.append(floatingPointLiteral.get(token))
      } else {
        sb.append(token)
      }
      sb.append(" ")
    })
    sb.toString()
  }

  private def importMaps(mapFile: String): Unit = {
    val lines = try {
      Files.readAllBytes(Paths.get(mapFile))
    } catch {
      case e:IOException => e.printStackTrace()
    }
  }

  private def fillMap(map:mutable.HashMap[String,String], keys:String, values:String) = {
    if (!keys.isEmpty) {
      val keysSplitted = keys.split(",")
      val valuesSplitted = values.split(",")
      for (i <- keysSplitted.indices)
        map.put(valuesSplitted(i), keysSplitted(i))
    }
  }

  private def fillStringMap(map:mutable.HashMap[String,String], keys:String, values:String) = {
    if (!keys.isEmpty){
      val keysSplitted = keys.substring(1).split("\",\"")
      val valuesSplitted = values.split(",")
      for(i  <- keysSplitted.indices)
        map.put(valuesSplitted(i), "\""+keysSplitted(i)+"\"")
    }
  }
}
