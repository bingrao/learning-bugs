package org.ucf.ml
package tree

import com.github.javaparser.ast.body._
import com.github.javaparser.ast._
import com.github.javaparser.ast.`type`._
import com.github.javaparser.ast.body.Parameter
import com.github.javaparser.ast.expr.{FieldAccessExpr, MethodCallExpr, Name, SimpleName}
import com.github.javaparser.ast.stmt._
import com.github.javaparser.ast.body.VariableDeclarator
import com.github.javaparser.ast.expr._
import scala.collection.JavaConversions._

trait EnrichedTrees extends utils.Common {

  implicit class genCompilationUnit(node:CompilationUnit) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      // 1. package declaration
      val package_decl = node.getPackageDeclaration
      if (package_decl.isPresent) package_decl.get().genCode(ctx, numsIntent)

      // 2. Import Statements
      node.getImports.foreach(impl => {
        impl.genCode(ctx, numsIntent)
        ctx.appendNewLine()
      })

      // 3. A list of defined types, such as Class, Interface, Enum, Annotation ...
      node.getTypes.foreach(typeDecl => typeDecl.genCode(ctx, numsIntent))

    }
  }

  implicit class genPackageDeclaration(node: PackageDeclaration) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("package") else "package")

      node.getName.genCode(ctx, numsIntent)
      ctx.append(";")
      ctx.appendNewLine()
    }
  }

  /**
   *  import java.io.*;
   *  import static java.io.file.out;
   * @param node
   */
  implicit class genImportDeclaration(node:ImportDeclaration) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {

      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("import") else "import")

      if (node.isStatic) ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("static") else "static")

      node.getName.genCode(ctx, numsIntent)

      if (node.isAsterisk) {
        ctx.append(".")
        ctx.append("*")
      }
      ctx.append(";")
      ctx.appendNewLine()
    }
  }

  implicit class genTypeDeclaration(node:TypeDeclaration[_]) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      node match {
        /**
         *  TypeDeclaration
         *     -- EnumDeclaration
         *     -- AnnotationDeclaration
         *     -- ClassOrInterfaceDeclaration
         */
        case n: EnumDeclaration => {n.genCode(ctx, numsIntent)}
        case n: AnnotationDeclaration => {n.genCode(ctx, numsIntent)}
        case n: ClassOrInterfaceDeclaration => {n.genCode(ctx, numsIntent)}
      }
      
    }
  }

  implicit class genEnumDeclaration(node:EnumDeclaration) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {

      val modifier = node.getModifiers
      modifier.foreach(_.genCode(ctx, numsIntent))

      node.getName.genCode(ctx, numsIntent)

      ctx.append("{")
      val entries = node.getEntries
      entries.foreach(entry => {
        entry.genCode(ctx, numsIntent)
        if (entry != entries.last) ctx.append(",")
      })
      ctx.append("}")
      ctx.appendNewLine()
    }
  }

  implicit class genAnnotationDeclaration(node:AnnotationDeclaration) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      //TODO, No implementation about annotation
      ctx.append(node.toString, numsIntent)
    }
  }

  implicit class genClassOrInterfaceDeclaration(node:ClassOrInterfaceDeclaration) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      // 1. Class Modifiers, such as public/private
      val modifiers = node.getModifiers
      modifiers.foreach(modifier => modifier.genCode(ctx, numsIntent))

      if (node.isInterface)
        ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("interface") else "interface")
      else
        ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("class") else "class")

      // 2. Interface/Class Name
      if (ctx.isAbstract)
        ctx.append(ctx.type_maps.getNewContent(node.getNameAsString))
      else
        node.getName.genCode(ctx)

      // 3. type parameters public interface Predicate<T> {}
      val tps = node.getTypeParameters
      tps.foreach(_.genCode(ctx, numsIntent))

      ctx.append("{")
      ctx.appendNewLine()

      // 3. Class Members: Filed and method, constructor
      val members = node.getMembers
      members.foreach(bodyDecl => bodyDecl.genCode(ctx, numsIntent))

      ctx.append("}")
      ctx.appendNewLine()
      
    }
  }

  implicit class genBodyDeclaration(node:BodyDeclaration[_]){
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      //TODO, only implment [[CallableDeclaration]] to handle with Method Declare
      node match {
        case n: InitializerDeclaration => {n.genCode(ctx, numsIntent)}
        case n: FieldDeclaration => {n.genCode(ctx, numsIntent)}
        case n: TypeDeclaration[_] => n.genCode(ctx, numsIntent)
        case n: EnumConstantDeclaration => {n.genCode(ctx, numsIntent)}
        case n: AnnotationMemberDeclaration => {n.genCode(ctx, numsIntent)}
        case n: CallableDeclaration[_] => {n.genCode(ctx, numsIntent)}
      }
    }
  }

  implicit class genInitializerDeclaration(node:InitializerDeclaration) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      //TODO, no implmentation for learning bugs
      ctx.append(node.toString(), numsIntent)
    }
  }

  implicit class genFieldDeclaration(node:FieldDeclaration) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {

      node.getModifiers.foreach(_.genCode(ctx, numsIntent))

      val varibles = node.getVariables
      varibles.foreach(ele => {
        if (ele == varibles.head)
            ele.getType.genCode(ctx)

        ele.getName.genCode(ctx)

        if (ele.getInitializer.isPresent){
          ctx.append("=")
          ele.getInitializer.get().genCode(ctx)
        }

        if (ele != node.getVariables.last) ctx.append(",")
      })
      ctx.append(";")
      ctx.appendNewLine()
    }
  }

  implicit class genEnumConstantDeclaration(node:EnumConstantDeclaration) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      //TODO, no implmentation for learning bugs
      ctx.append(node.toString, numsIntent)
    }
  }

  implicit class genAnnotationMemberDeclaration(node:AnnotationMemberDeclaration) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      //TODO, no implmentation for learning bugs
      ctx.append(node.toString, numsIntent)
    }
  }

  implicit class genCallableDeclaration(node:CallableDeclaration[_]){
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      node match {
        //TODO, no implmentation [[ConstructorDeclaration]] for learning bugs
        case n: ConstructorDeclaration => {n.genCode(ctx, numsIntent)}
        case n: MethodDeclaration => {n.genCode(ctx, numsIntent)}
      }
    }
  }

  implicit class genConstructorDeclaration(node:ConstructorDeclaration) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      //TODO, no implmentation for learning bugs
      ctx.append(node.toString, numsIntent)
    }
  }

  implicit class genMethodDeclaration(node:MethodDeclaration) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {

      /*modifiers, such as public*/
      val modifiers = node.getModifiers
      modifiers.foreach(modifier => modifier.genCode(ctx, numsIntent))

      /*Method return type, such as void, int, string*/
      node.getType.genCode(ctx, numsIntent)

      /*Method name, such as hello*/
      if (ctx.isAbstract)
        ctx.append(ctx.method_maps.getNewContent(node.getName.toString()))
      else
        node.getName.genCode(ctx)

      /*formal paramters*/
      ctx.append("(")
      val parameters = node.getParameters
      parameters.foreach(p => {
        p.genCode(ctx, numsIntent)
        if (p != parameters.last) ctx.append(",")
      })
      ctx.append(")")

      /*Method Body*/
      val body = node.getBody
      if (body.isPresent) body.get().genCode(ctx, numsIntent)
    }
  }


  /************************** Statement ***************************/

  implicit class genStatement(node:Statement) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      node match {
        case n:ForEachStmt => n.genCode(ctx, numsIntent)
        case n:LocalClassDeclarationStmt => n.genCode(ctx, numsIntent)
        case n:ContinueStmt => n.genCode(ctx, numsIntent)
        case n:ExpressionStmt => n.genCode(ctx, numsIntent)
        case n:LabeledStmt => n.genCode(ctx, numsIntent)
        case n:YieldStmt => n.genCode(ctx, numsIntent)
        case n:ReturnStmt => n.genCode(ctx, numsIntent)
        case n:WhileStmt => n.genCode(ctx, numsIntent)
        case n:EmptyStmt => n.genCode(ctx, numsIntent)
        case n:UnparsableStmt => n.genCode(ctx, numsIntent)
        case n:IfStmt => n.genCode(ctx, numsIntent)
        case n:BreakStmt => n.genCode(ctx, numsIntent)
        case n:AssertStmt => n.genCode(ctx, numsIntent)
        case n:ExplicitConstructorInvocationStmt => n.genCode(ctx, numsIntent)
        case n:DoStmt => n.genCode(ctx, numsIntent)
        case n:ForStmt => n.genCode(ctx, numsIntent)
        case n:ThrowStmt => n.genCode(ctx, numsIntent)
        case n:TryStmt => n.genCode(ctx, numsIntent)
        case n:SwitchStmt => n.genCode(ctx, numsIntent)
        case n:SynchronizedStmt => n.genCode(ctx, numsIntent)
        case n:BlockStmt => n.genCode(ctx, numsIntent)
      }

    }
  }

  implicit class genForEachStmt(node:ForEachStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val integral = node.getIterable
      val variable = node.getVariable
      val body = node.getBody

      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("for") else "for")
      ctx.append("(")
      variable.genCode(ctx)
      ctx.append(":")
      integral.genCode(ctx)
      ctx.append(")")
      body.genCode(ctx)
    }
  }

  implicit class genLocalClassDeclarationStmt(node:LocalClassDeclarationStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      node.getClassDeclaration.genCode(ctx)
    }
  }

  implicit class genContinueStmt(node:ContinueStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      val label =  node.getLabel
      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("continue") else "continue")
      if (label.isPresent) label.get().genCode(ctx, numsIntent)
      ctx.append(";")
      ctx.appendNewLine()
    }
  }

  implicit class genExpressionStmt(node:ExpressionStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      node.getExpression.genCode(ctx, numsIntent)
      if (!node.getParentNode.get().isInstanceOf[Expression]) {
        ctx.append(";")
        ctx.appendNewLine()
      }
    }
  }

  implicit class genLabeledStmt(node:LabeledStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      val label = node.getLabel
      val sts = node.getStatement

      label.genCode(ctx, numsIntent)
      ctx.append(":")
      sts.genCode(ctx, numsIntent)

      ctx.append(";")
      ctx.appendNewLine()

    }
  }

  implicit class genYieldStmt(node:YieldStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("yield") else "yield")
      node.getExpression.genCode(ctx, numsIntent)
      ctx.append(";")
      ctx.appendNewLine()
    }
  }

  implicit class genReturnStmt(node:ReturnStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("return") else "return")
      val expr = node.getExpression
      if (expr.isPresent) expr.get().genCode(ctx, numsIntent)
      ctx.append(";")
      ctx.appendNewLine()
    }
  }

  implicit class genWhileStmt(node:WhileStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      val body = node.getBody
      val condition = node.getCondition
      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("while") else "while")
      ctx.append("(")
      condition.genCode(ctx, numsIntent)
      ctx.append(")")
      ctx.appendNewLine()
      body.genCode(ctx, numsIntent)
    }
  }

  implicit class genEmptyStmt(node:EmptyStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      ctx.append(";")
      ctx.appendNewLine()
    }
  }

  implicit class genUnparsableStmt(node:UnparsableStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      //TODO, not for this project
      ctx.append(node.toString, numsIntent)
    }
  }

  implicit class genIfStmt(node:IfStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      val condition = node.getCondition
      val thenStmt = node.getThenStmt
      val elseStmt = node.getElseStmt

      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("if") else "if")
      ctx.append("(")
      condition.genCode(ctx, numsIntent)
      ctx.append(")")

      ctx.appendNewLine()
      thenStmt.genCode(ctx, numsIntent)

      if (elseStmt.isPresent){
        ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("else") else "else")
        ctx.appendNewLine()
        elseStmt.get().genCode(ctx, numsIntent)
      }
    }
  }

  implicit class genBreakStmt(node:BreakStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      val label = node.getLabel
      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("break") else "break")
      if (label.isPresent) label.get().genCode(ctx, numsIntent)

      ctx.append(";")
      ctx.appendNewLine()
    }
  }

  implicit class genAssertStmt(node:AssertStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      val check = node.getCheck
      val msg = node.getMessage

      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("assert") else "assert")
      check.genCode(ctx, numsIntent)

      if (msg.isPresent) {
        ctx.append(":")
        msg.get().genCode(ctx, numsIntent)
      }

      ctx.append(";")
      ctx.appendNewLine()
    }
  }

  implicit class genExplicitConstructorInvocationStmt(node:ExplicitConstructorInvocationStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      //TODO, not for this project
      ctx.append(node.toString, numsIntent)
    }
  }

  implicit class genDoStmt(node:DoStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      val body = node.getBody
      val condition = node.getCondition

      body.genCode(ctx, numsIntent)
      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("while") else "while")
      ctx.append("(")
      condition.genCode(ctx, numsIntent)
      ctx.append(")")

      ctx.append(";")
      ctx.appendNewLine()
    }
  }

  implicit class genForStmt(node:ForStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("for") else "for")
      ctx.append("(")
      val initial = node.getInitialization
      initial.foreach(init => {
        init.genCode(ctx, numsIntent)
        if (init != initial.last) ctx.append(",")
      })
      ctx.append(";")

      val compare = node.getCompare
      if (compare.isPresent) compare.get().genCode(ctx, numsIntent)

      ctx.append(";")

      val update = node.getUpdate
      update.foreach(up => {
        up.genCode(ctx, numsIntent)
        if (up != update.last) ctx.append(",")
      })
      ctx.append(")")

      val body = node.getBody
      body.genCode(ctx, numsIntent)
    }
  }

  implicit class genThrowStmt(node:ThrowStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("throw") else "throw")
      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("new") else "new")
      node.getExpression.genCode(ctx)
      ctx.appendNewLine()
    }
  }

  implicit class genTryStmt(node:TryStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      //TODO not for this project
      val tryResources = node.getResources
      val tryCatch = node.getCatchClauses
      val tryFinally = node.getFinallyBlock
      val tryBlock = node.getTryBlock

      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("try") else "try")
      if (tryResources.size() != 0){
        ctx.append("(")
        tryResources.foreach(expr => {
          expr.genCode(ctx, numsIntent)
          if (expr != tryResources.last) ctx.append(",")
        })
        ctx.append(")")
      }

      tryBlock.genCode(ctx)

      tryCatch.foreach(_.genCode(ctx, numsIntent))

      if (tryFinally.isPresent) tryFinally.get().genCode(ctx, numsIntent)
    }
  }

  implicit class genCatchClause(node:CatchClause) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      val parameter = node.getParameter
      val body = node.getBody
      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("catch") else "catch")
      ctx.append("(")
      parameter.genCode(ctx, numsIntent)
      ctx.append(")")
      body.genCode(ctx)
      ctx.appendNewLine()
    }
  }


  implicit class genSwitchStmt(node:SwitchStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {

      val entries = node.getEntries
      val selector = node.getSelector
      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("switch") else "switch")

      ctx.append("(")
      selector.genCode(ctx, numsIntent)
      ctx.append(")")
      ctx.appendNewLine()

      ctx.append("{")
      entries.foreach(_.genCode(ctx))
      ctx.append("}")
      ctx.appendNewLine()
    }
  }

  implicit class genSynchronizedStmt(node:SynchronizedStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("synchronized") else "synchronized")
      ctx.append("(")
      node.getExpression.genCode(ctx, numsIntent)
      ctx.append(")")
      node.getBody.genCode(ctx)
    }
  }

  implicit class genBlockStmt(node:BlockStmt) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit  = {
      ctx.append("{")
      ctx.appendNewLine()
      node.getStatements.foreach(sts => sts.genCode(ctx, numsIntent))
      ctx.append("}")
      ctx.appendNewLine()
    }
  }


  /******************************* Expression ********************************/
  implicit class genExpression(node:Expression){
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      node match {
        case expr:ArrayAccessExpr  => expr.genCode(ctx, numsIntent)
        case expr:ClassExpr  => expr.genCode(ctx, numsIntent)
        case expr:LambdaExpr  => expr.genCode(ctx, numsIntent)
        case expr:ArrayCreationExpr  => expr.genCode(ctx, numsIntent)
        case expr:ConditionalExpr  => expr.genCode(ctx, numsIntent)
        case expr:MethodCallExpr  => expr.genCode(ctx, numsIntent)
        case expr:AnnotationExpr  => expr.genCode(ctx, numsIntent)
        case expr:AssignExpr  => expr.genCode(ctx, numsIntent)
        case expr:InstanceOfExpr  => expr.genCode(ctx, numsIntent)
        case expr:ThisExpr  => expr.genCode(ctx, numsIntent)
        case expr:NameExpr  => expr.genCode(ctx, numsIntent)
        case expr:CastExpr  => expr.genCode(ctx, numsIntent)
        case expr:MethodReferenceExpr  => expr.genCode(ctx, numsIntent)
        case expr:EnclosedExpr  => expr.genCode(ctx, numsIntent)
        case expr:VariableDeclarationExpr  => expr.genCode(ctx, numsIntent)
        case expr:SwitchExpr  => expr.genCode(ctx, numsIntent)
        case expr:LiteralExpr => expr.genCode(ctx, numsIntent)
        case expr:ObjectCreationExpr  => expr.genCode(ctx, numsIntent)
        case expr:SuperExpr  => expr.genCode(ctx, numsIntent)
        case expr:UnaryExpr  => expr.genCode(ctx, numsIntent)
        case expr:BinaryExpr  => expr.genCode(ctx, numsIntent)
        case expr:FieldAccessExpr  => expr.genCode(ctx, numsIntent)
        case expr:TypeExpr  => expr.genCode(ctx, numsIntent)
        case expr:ArrayInitializerExpr  => expr.genCode(ctx, numsIntent)
      }
    }
  }


  implicit class genArrayAccessExpr(node:ArrayAccessExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val name = node.getName
      val index = node.getIndex
      name.genCode(ctx, numsIntent)
      ctx.append("[")
      index.genCode(ctx, numsIntent)
      ctx.append("]")
    }
  }


  implicit class genClassExpr(node:ClassExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit= {

      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("Object") else "Object")
      ctx.append(".")
      node.getType.genCode(ctx)
      ctx.appendNewLine()
    }
  }


  implicit class genLambdaExpr(node:LambdaExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {

      val parameters = node.getParameters

      if (parameters.size > 1)
        ctx.append("(")

      parameters.foreach(p => {
        p.genCode(ctx, numsIntent)
        if (p != parameters.last) ctx.append(",")
      })

      if (parameters.size > 1)
        ctx.append(")")

      ctx.append("->")

      val body = node.getBody

      body.genCode(ctx, numsIntent)

    }
  }


  implicit class genArrayCreationExpr(node:ArrayCreationExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {

      val eleType = node.getElementType
      val initial = node.getInitializer
      val levels = node.getLevels

      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("new") else "new")
      eleType.genCode(ctx, numsIntent)
      for (level <- levels){
        ctx.append("[")
        val dim = level.getDimension
        if (dim.isPresent) dim.get().genCode(ctx, numsIntent)
        ctx.append("]")
      }

      if (initial.isPresent) initial.get().genCode(ctx, numsIntent)
    }
  }

  implicit class genConditionalExpr(node:ConditionalExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val condition = node.getCondition
      val thenExpr = node.getThenExpr
      val elseExpr = node.getElseExpr

      condition.genCode(ctx, numsIntent)
      ctx.append("?")
      thenExpr.genCode(ctx, numsIntent)
      ctx.append(":")
      elseExpr.genCode(ctx, numsIntent)
    }
  }

  implicit class genMethodCallExpr(node:MethodCallExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {

      val scope = node.getScope
      val arguments = node.getArguments

      if (scope.isPresent) {

        if (ctx.isAbstract) {
          val scope_value = ctx.variable_maps.getNewContent(scope.get().toString)
          ctx.append(scope_value)
        } else
          scope.get().genCode(ctx, numsIntent)
        ctx.append(".")
      }
      if (ctx.isAbstract) {
        val funcName = ctx.method_maps.getNewContent(node.getName.asString())
        ctx.append(funcName)
      } else node.getName.genCode(ctx)

      ctx.append("(")
      arguments.foreach(expr => {
        expr.genCode(ctx, numsIntent)
        if (expr != arguments.last) ctx.append(",")
      })
      ctx.append(")")
    }
  }

  implicit class genAnnotationExpr(node:AnnotationExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      //TODO, not for this projects
      ctx.append(node.toString)
    }
  }

  implicit class genAssignExpr(node:AssignExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit= {
      val left = node.getTarget
      val right = node.getValue
      val op = node.getOperator

      left.genCode(ctx, numsIntent)

      ctx.append(op.asString())

      right.genCode(ctx, numsIntent)


    }
  }

  implicit class genInstanceOfExpr(node:InstanceOfExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      node.getExpression.genCode(ctx)
      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("instanceof") else "instanceof")
      node.getType.genCode(ctx)
    }
  }

  implicit class genThisExpr(node:ThisExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("This") else "This")
    }
  }

  implicit class genNameExpr(node:NameExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      if (ctx.isAbstract) {
        ctx.append(ctx.variable_maps.getNewContent(node.getName.asString()))
      } else node.getName.genCode(ctx)
    }
  }

  implicit class genCastExpr(node:CastExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      //TODO no implement for this project
      ctx.append(node.toString, numsIntent)
    }
  }

  implicit class genMethodReferenceExpr(node:MethodReferenceExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val ident = node.getIdentifier
      val scope = node.getScope

      scope.genCode(ctx, numsIntent)
      ctx.append("::")
      ctx.append(ident)
    }
  }

  implicit class genEnclosedExpr(node:EnclosedExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit= {
      ctx.append("(")
      node.getInner.genCode(ctx)
      ctx.append(")")
    }
  }

  implicit class genVariableDeclarationExpr(node:VariableDeclarationExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      node.getModifiers.foreach(_.genCode(ctx, numsIntent))
      val varibles = node.getVariables
      varibles.foreach(_.genCode(ctx, numsIntent))
    }
  }

  implicit class genSwitchExpr(node:SwitchExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("switch") else "switch")
      ctx.append("(")
      node.getSelector.genCode(ctx)
      ctx.append(")")
      ctx.append("{")

      node.getEntries.foreach(_.genCode(ctx))

      ctx.append("}")
      ctx.appendNewLine()
    }
  }

  implicit class genSwitchEntry(node:SwitchEntry) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val tp = node.getType.asInstanceOf[Int]
      val lables = node.getLabels
      val sts = node.getStatements

      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("case") else "case")

      lables.foreach(expr => {
        expr.genCode(ctx)
        if (expr != lables.last) ctx.append(",")
      })

      ctx.append("->")

      ctx.append("{")
      sts.foreach(_.genCode(ctx))
      ctx.append("}")

    }
  }

  // subclass
  implicit class genLiteralExpr(node:LiteralExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      node match {
        case expr:NullLiteralExpr => ctx.append(expr.toString)
        case expr:BooleanLiteralExpr => ctx.append(expr.getValue.toString)
        case expr:LiteralStringValueExpr  => expr.genCode(ctx, numsIntent)
      }
    }
  }

  implicit class genLiteralStringValueExpr(node:LiteralStringValueExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      node match {
        case expr: TextBlockLiteralExpr => expr.genCode(ctx, numsIntent)
        case expr: CharLiteralExpr => expr.genCode(ctx, numsIntent)
        case expr: DoubleLiteralExpr => expr.genCode(ctx, numsIntent)
        case expr: LongLiteralExpr => expr.genCode(ctx, numsIntent)
        case expr: StringLiteralExpr => expr.genCode(ctx, numsIntent)
        case expr: IntegerLiteralExpr => expr.genCode(ctx, numsIntent)
      }
    }
  }

  implicit class genTextBlockLiteralExpr(node:TextBlockLiteralExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val value = if (ctx.isAbstract) ctx.textBlock_maps.getNewContent(node.getValue) else node.asString()
      ctx.append(value)
    }
  }

  implicit class genCharLiteralExpr(node:CharLiteralExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val value = if (ctx.isAbstract) ctx.char_maps.getNewContent(node.getValue) else node.asChar().toString
      ctx.append(value)
    }
  }

  implicit class genDoubleLiteralExpr(node:DoubleLiteralExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val value = if (ctx.isAbstract)  ctx.double_maps.getNewContent(node.getValue) else node.asDouble().toString
      ctx.append(value)
    }
  }

  implicit class genLongLiteralExpr(node:LongLiteralExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val value = if (ctx.isAbstract)  ctx.long_maps.getNewContent(node.getValue) else node.asNumber().toString
      ctx.append(value)
    }
  }

  implicit class genStringLiteralExpr(node:StringLiteralExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val value = if (ctx.isAbstract)  ctx.string_maps.getNewContent(node.getValue) else node.asString()
      ctx.append("\"" + value + "\"")
    }
  }

  implicit class genIntegerLiteralExpr(node:IntegerLiteralExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val value = if (ctx.isAbstract)  ctx.int_maps.getNewContent(node.getValue) else node.asNumber().toString
      ctx.append(value)
    }
  }

  /**
   *  new B().new C();
   *  scope --> new B()
   *  type --> new C()
   * @param node
   */
  implicit class genObjectCreationExpr(node:ObjectCreationExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val arguments = node.getArguments
      val scope = node.getScope
      val tp = node.getType

      if (scope.isPresent) {
        scope.get().genCode(ctx, numsIntent)
        ctx.append(".")
      }

      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("new") else "new")

      tp.genCode(ctx, numsIntent)

      ctx.append("(")
      arguments.foreach(expr =>{
        expr.genCode(ctx, numsIntent)
        if (expr != arguments.last) ctx.append(",")
      })
      ctx.append(")")
    }
  }


  implicit class genSuperExpr(node:SuperExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {

      val tpName = node.getTypeName

      if (tpName.isPresent){
        tpName.get().genCode(ctx)
        ctx.append(".")
      }
      ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("super") else "super")
    }
  }

  implicit class genUnaryExpr(node:UnaryExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val op = node.getOperator.asString()
      val expr = node.getExpression
      if (node.isPostfix) {
        expr.genCode(ctx, numsIntent)
        ctx.append(op)
      }
      if (node.isPrefix) {
        ctx.append(op)
        expr.genCode(ctx, numsIntent)
      }
    }
  }

  implicit class genBinaryExpr(node:BinaryExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val left = node.getLeft
      val op = node.getOperator.asString()
      val right = node.getRight
      left.genCode(ctx, numsIntent)
      ctx.append(op)
      right.genCode(ctx, numsIntent)
    }
  }

  implicit class genFieldAccessExpr(node:FieldAccessExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {

      if (ctx.isAbstract) {
        val scope_value = ctx.variable_maps.getNewContent(node.getScope.toString)
        ctx.append(scope_value)
      } else
        node.getScope.genCode(ctx, numsIntent)

      ctx.append(".")

      // filed
      if (ctx.isAbstract) {
        val name = ctx.variable_maps.getNewContent(node.getName.asString())
        ctx.append(name)
      } else
        node.getName.genCode(ctx)
    }
  }

  implicit class genTypeExpr(node:TypeExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      node.getType.genCode(ctx, numsIntent)
    }
  }

  implicit class genArrayInitializerExpr(node:ArrayInitializerExpr) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val values = node.getValues
      ctx.append("{")
      values.foreach(ele => {
        ele.genCode(ctx, numsIntent)
        if (ele != values.last) ctx.append(",")
      })
      ctx.append("}")
    }
  }

  /******************************* Variable ********************************/
  implicit class genVariableDeclarator(node: VariableDeclarator) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val tp = node.getType
      val name = node.getName
      val init = node.getInitializer

      tp.genCode(ctx, numsIntent)

      if (ctx.isAbstract) {
        val value =  ctx.variable_maps.getNewContent(name.asString())
        ctx.append(value)
      } else name.genCode(ctx)


      if (init.isPresent){
        ctx.append("=")
        init.get().genCode(ctx, numsIntent)
      }
    }
  }

  /******************************* Node ************************/

  implicit class addPosition(node:Node) {
    def getPosition(ctx:Context, numsIntent:Int=0) = ctx.getNewPosition
  }

  implicit class genSimpleName(node:SimpleName) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      ctx.append(node.getIdentifier)

    }
  }

  implicit class genModifier(node: Modifier) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      ctx.append(node.getKeyword.asString())

    }
  }

  implicit class genType(node:Type) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {

      node match {
        case tp:UnionType  =>{
          val value = if (ctx.isAbstract) ctx.type_maps.getNewContent(node.asString()) else node.asString()
          ctx.append(value)
        }
        case tp:VarType  =>{
          val value = if (ctx.isAbstract) ctx.type_maps.getNewContent(node.asString()) else node.asString()
          ctx.append(value)
        }
        case tp:ReferenceType  => tp.genCode(ctx, numsIntent)
        case tp:UnknownType  =>{
          val value = if (ctx.isAbstract) ctx.type_maps.getNewContent(node.asString()) else node.asString()
          ctx.append(value)
        }
        case tp:PrimitiveType  =>{
          val value = if (ctx.isAbstract) ctx.type_maps.getNewContent(node.asString()) else node.asString()
          ctx.append(value)
        }
        case tp:WildcardType  =>{
          val value = if (ctx.isAbstract) ctx.type_maps.getNewContent(node.asString()) else node.asString()
          ctx.append(value)
        }
        case tp:VoidType  =>{
          val value = if (ctx.isAbstract) ctx.type_maps.getNewContent(node.asString()) else node.asString()
          ctx.append(value)
        }
        case tp:IntersectionType  =>{
          val value = if (ctx.isAbstract) ctx.type_maps.getNewContent(node.asString()) else node.asString()
          ctx.append(value)
        }
      }
    }
  }

  implicit class genReferenceType(node:ReferenceType) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      node match {
        case tp: ArrayType => tp.genCode(ctx, numsIntent)
        case tp: TypeParameter => tp.genCode(ctx, numsIntent)
        case tp: ClassOrInterfaceType => tp.genCode(ctx, numsIntent)
      }
    }
  }

  /**
   * So, int[][] becomes ArrayType(ArrayType(int)).
   * @param node
   */
  implicit class genArrayType(node:ArrayType) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      //TODO, need more details about
      val origin = node.getOrigin
      val comType = node.getComponentType
      comType.genCode(ctx, numsIntent)
      ctx.append("[")
      ctx.append("]")
    }
  }

  implicit class genTypeParameter(node:TypeParameter) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val name = node.getName
      val typeBound = node.getTypeBound

      ctx.append("<")
      name.genCode(ctx, numsIntent)
      if (typeBound.size() != 0){
        ctx.append(if (ctx.isAbstract) ctx.ident_maps.getNewContent("extends") else "extends")
        typeBound.foreach(bound => {
          bound.genCode(ctx, numsIntent)
          if (bound != typeBound.last) ctx.append("&")
        })
      }
      ctx.append(">")
    }
  }

  implicit class genClassOrInterfaceType(node:ClassOrInterfaceType) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val scope = node.getScope
      val name = node.getName
      val tps = node.getTypeArguments

      if (scope.isPresent){
        scope.get().genCode(ctx, numsIntent)
        ctx.append(".")
      }
      name.genCode(ctx, numsIntent)

      if (tps.isPresent){
        ctx.append("<")
        tps.get().foreach(_.genCode(ctx, numsIntent))
        ctx.append(">")
      }
    }
  }


  implicit class genParameter(node:Parameter) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val tp = node.getType
      val name = node.getName
      tp.genCode(ctx, numsIntent)

      if (ctx.isAbstract) {
        val value =  ctx.variable_maps.getNewContent(name.asString())
        ctx.append(value)
      } else
        name.genCode(ctx, numsIntent)
    }
  }

  implicit class genName(node:Name) {
    def genCode(ctx:Context, numsIntent:Int=0):Unit = {
      val qualifier = node.getQualifier
      if (qualifier.isPresent){
        qualifier.get().genCode(ctx, numsIntent)
        ctx.append(".")
      }
      val name = node.getIdentifier
      ctx.append(name)
    }
  }

  def expand_scope(ctx:Context, scope:Node):Unit = {
    scope match {
      case expr: MethodCallExpr => {
        val expr_name = expr.getName
        val expr_scope = expr.getScope

        // method name
        if (ctx.isAbstract) {
          ctx.method_maps.getNewContent(expr_name.asString())
        } else
          expr_name.genCode(ctx)

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
