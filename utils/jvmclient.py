from py4j.java_gateway import JavaGateway, GatewayParameters

class JVMClient:
	def __init__(self, ctx=None, port=18888):
		self.gateway = JavaGateway(gateway_parameters=GatewayParameters(port=port))
		self.javaParser = self.geteway.entry_point
	def get_abstract_code(self, inputPath=None, idiomPath=None, granularity="class", isFile=True):
		assert granularity != "class" or granularity != "method"
		if granularity == "class":
			if isFile:
				abstract_code = self.javaParser.getAbstractCodeFromFileSClass(inputPath, idiomPath)
			else:
				abstract_code = self.javaParser.getAbstractCodeFromStringClass(inputPath, idiomPath)
		else:
			if isFile:
				abstract_code = self.javaParser.getAbstractCodeFromFileMethod(inputPath, idiomPath)
			else:
				abstract_code = self.javaParser.getAbstractCodeFromStringMethod(inputPath, idiomPath)

		return list(abstract_code)