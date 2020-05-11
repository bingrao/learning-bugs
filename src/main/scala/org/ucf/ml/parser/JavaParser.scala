package org.ucf.ml
package parser

import com.github.javaparser.ast.`type`.Type
import com.github.javaparser.{JavaToken, StaticJavaParser}
import com.github.javaparser.ast.CompilationUnit
import com.github.javaparser.ast.body.MethodDeclaration
import com.github.javaparser.ast.expr.{MethodCallExpr, MethodReferenceExpr}

import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer
import java.util.stream.Collectors

class JavaParser extends utils.context with Visitor{

  /**
   * printAST("./log/UnionMain.Yaml", cu)
   * printAST("./log/UnionMain.Xml", cu, "xml")
   * printAST("./log/UnionMain.dot", cu, "dot")
   *
   * @param path
   * @param cu
   * @param format
   */
  def printAST(path:String=null, cu: CompilationUnit, format:String = "ymal") = try {
    if (path == null){
      logger.info("The input source code is: ")
      println(cu.toString)
    } else {
      import com.github.javaparser.printer.{DotPrinter, XmlPrinter, YamlPrinter}
      val context = format match {
        case "xml" => (new XmlPrinter(true)).output(cu)
        case "dot" => (new DotPrinter(true)).output(cu)
        case "ymal" | _ => (new YamlPrinter(true)).output(cu)
      }
      write(path, context = cu.toString() + context)
    }
  } catch {
    case e:Exception =>{
      logger.error(f"Write file ${path} failed in the format of ${format}")}
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
  def getMethodCall(cu:CompilationUnit) = {
    val collector = ListBuffer[MethodCallExpr]()
    MethodCallCollector().visit(cu, collector)
    collector
  }
  def getMethodCall_(cu:CompilationUnit) =
    cu.findAll(classOf[MethodCallExpr])
      .stream()
      .collect(Collectors.toList[MethodCallExpr]())
      .toList

  def getMethodDecl(cu:CompilationUnit) = {
    val collector = ListBuffer[MethodDeclaration]()
    MethodDeclCollector().visit(cu, collector)
    collector
  }
  def getMethodDecl_(cu:CompilationUnit) =
    cu.findAll(classOf[MethodDeclaration])
      .stream()
      .collect(Collectors.toList[MethodDeclaration]())
      .toList

  def getMethodRef(cu:CompilationUnit) = {
    val collector = ListBuffer[MethodReferenceExpr]()
    MethodRefCollector().visit(cu, collector)
    collector
  }
  def getMethodRef_(cu:CompilationUnit) =
    cu.findAll(classOf[MethodReferenceExpr])
      .stream()
      .collect(Collectors.toList[MethodReferenceExpr]())
      .toList

  def getTypes(cu:CompilationUnit) =
    cu.findAll(classOf[Type])
      .stream()
      .collect(Collectors.toList[Type]())
      .toList

}

object JavaParser extends JavaParser {
  def main(args: Array[String]): Unit = {
    /**
     * https://javaparser.org/inspecting-an-ast/
     */
    val cu = getComplationUnit("src/main/java/org/ucf/ml/JavaApp.java", "class")
    getMethodCall_(cu).foreach(println _)
    println("********************************************")
    getTypes(cu).foreach(println _)
    println("********************************************")
    getMethodRef(cu).foreach(println _)
  }
}
