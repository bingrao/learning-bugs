package org.ucf.ml
package plaintext

import java.io.IOException
import java.nio.file.Files
import java.nio.file.Paths
class SourceCodeAnalyzer {
  def readSourceCode(filePath:String) = {
    var reg:String = ""
    try {
      reg = new String(Files.readAllBytes(Paths.get(filePath)))
    } catch {
      case e:IOException => e.printStackTrace()
    }
    reg
  }
  def removeCommentsAndAnnotations(sourceCode:String) = sourceCode
    .replaceAll("(?:/\\*(?:[^*]|(?:\\*+[^*/]))*\\*+/)|(?://.*)","") //remove comments
    .replaceAll("@.+", "") //remove annotations
}
