private void success(io.netty.channel.Channel channel) {
    org.mycat.netty.mysql.MySQLHandshakeHandler.logger.info("success info return form MySQLHandshakeHandler");
    io.netty.buffer.ByteBuf out = channel.alloc().buffer();
    org.mycat.netty.mysql.OK ok = new org.mycat.netty.mysql.OK();
    ok.sequenceId = 2;
    ok.setStatusFlag(Flags.SERVER_STATUS_AUTOCOMMIT);
    out.writeBytes(ok.toPacket());
    channel.writeAndFlush(out);
}