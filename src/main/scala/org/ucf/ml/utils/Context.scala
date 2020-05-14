package org.ucf.ml
package utils

import java.util.concurrent.atomic.AtomicInteger

import com.github.javaparser.ast.`type`.ClassOrInterfaceType
import com.github.javaparser.ast.expr.{FieldAccessExpr, MethodCallExpr, NameExpr}
import com.github.javaparser.ast.Node

class Context extends Common {
  private val position_offset = new AtomicInteger()
  def getNewPosition = position_offset.getAndIncrement()
  def getCurrentPositionOffset = position_offset.get()

  private var current_target = "buggy"
  def getCurrentTarget = this.current_target
  def setCurrentTarget(target:String) = this.current_target = target

  private val buggy_abstract = new StringBuilder()
  private val fixed_abstract = new StringBuilder()

  def isAddPostion = false

  def attachePosition(content:String) = if (isAddPostion) f"${content}#${this.getNewPosition} " else f"${content} "

  def append(content:String) = this.getCurrentTarget match {
    case "buggy" => this.buggy_abstract.append(attachePosition(content))
    case "fixed" => this.fixed_abstract.append(attachePosition(content))
  }

  def isNewLine = true
  def appendNewLine(level:Int=0):Unit = this.getCurrentTarget match {
    case "buggy" => if (isNewLine) this.buggy_abstract.append("\n")
    case "fixed" => if (isNewLine) this.fixed_abstract.append("\n")
  }

  def get_buggy_abstract = buggy_abstract.toString()
  def get_fixed_abstract = fixed_abstract.toString()



  /*************************** set up and look up idioms ****************************/
  private val idioms = readIdioms("idioms/idioms.csv")

  def ident_maps = new Count[String, String]("Ident", idioms)

  val textBlock_maps = new Count[String, String]("text", idioms)
  val string_maps = new Count[String, String]("String", idioms)
  val char_maps = new Count[String, String]("Char", idioms)
  val int_maps = new Count[String, String]("Integer", idioms)
  val float_maps = new Count[String, String]("Float", idioms)
  val long_maps = new Count[String, String]("Long", idioms)
  val double_maps = new Count[String, String]("Double", idioms)

  val type_maps = new Count[String, String]("Type", idioms)
  val method_maps = new Count[String, String]("Method", idioms)
  val variable_maps = new Count[String, String]("Varl", idioms)


  def dumpy_mapping(path:String=null) = {
    textBlock_maps.dump_data(path)
    string_maps.dump_data(path)
    char_maps.dump_data(path)
    int_maps.dump_data(path)
    float_maps.dump_data(path)
    long_maps.dump_data(path)
    double_maps.dump_data(path)
    type_maps.dump_data(path)
    method_maps.dump_data(path)
    variable_maps.dump_data(path)
  }


  private val buffer = new StringBuilder()
  def prefix_buffer(content:String) = buffer.insert(0, content)
  def append_buffer(content:String) = buffer.append(content)
  def get_and_clear_buffer = {
    val reg = buffer.toString()
    buffer.clear()
    reg
  }

  def expand_scope(ctx:Context, scope:Node):Unit = {
    scope match {
      case expr: MethodCallExpr => {
        val expr_name = expr.getName
        val expr_scope = expr.getScope

        // method name
        ctx.method_maps.getNewContent(expr_name.asString())

        if (expr_scope.isPresent) expand_scope(ctx, expr_scope.get())
      }
      case tp:ClassOrInterfaceType => {
        val tp_name = tp.getName
        val tp_scope = tp.getScope
        if (tp_scope.isPresent) expand_scope(ctx, tp_scope.get())
      }
      case fd:FieldAccessExpr => {}
      case _ =>{}
    }
  }

}
