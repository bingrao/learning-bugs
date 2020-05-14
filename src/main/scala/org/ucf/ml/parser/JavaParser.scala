package org.ucf.ml
package parser


import com.github.javaparser.{JavaToken, StaticJavaParser}
import com.github.javaparser.ast.CompilationUnit
import scala.collection.JavaConversions._

class JavaParser(ctx:Context) extends Visitor  {

  /**
   * printAST("./log/UnionMain.Yaml", cu)
   * printAST("./log/UnionMain.Xml", cu, "xml")
   * printAST("./log/UnionMain.dot", cu, "dot")
   *
   * @param outPath
   * @param cu
   * @param format
   */

  def printAST(outPath:String=null, cu: CompilationUnit, format:String = "ymal") = try {
    if (outPath == null){
      logger.info("The input source code is: ")
      println(cu.toString)
    } else {
      import com.github.javaparser.printer.{DotPrinter, XmlPrinter, YamlPrinter}
      val context = format match {
        case "xml" => (new XmlPrinter(true)).output(cu)
        case "dot" => (new DotPrinter(true)).output(cu)
        case "ymal" | _ => (new YamlPrinter(true)).output(cu)
      }
      write(outPath, context = cu.toString() + context)
    }
  } catch {
    case e:Exception =>{
      logger.error(f"Write file ${outPath} failed in the format of ${format}")}
      e.printStackTrace()
  }

  def getComplationUnit(sourcePath:String, granularity:String) = {
    val source = Granularity.apply(sourcePath, granularity).getSourceCode()
    StaticJavaParser.parse(source)
  }

  def getTokens(cu:CompilationUnit) = cu.getTokenRange.get().toList

  def readTokens(filePath:String, granularity:String):List[JavaToken] = {
    val cu = this.getComplationUnit(filePath, granularity)
    this.getTokens(cu)
  }
}
