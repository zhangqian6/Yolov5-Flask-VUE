<template>
  <div id="Header">
    <div class="top-left-edition" >
      <span style="color: #67C23A; font-weight: bold;margin-right: 20px">
        <i class="el-icon-star-off" style="font-size: 23px">我的公众号：//可达鸭的深度学习教程</i>
      </span>
      <span>
        <i class="el-icon-time" style="font-size: 23px">CSDN、B站：BIT可达鸭</i>
      </span>
    </div>
    <div class="top-left-edition" >
      <h1>{{ msg }}</h1>
    </div>
  </div>
</template>
<script>
export default {
  name: "Header",
  data() {
    return {
      msg: "YOLOv5-WEB端dd",
      activeIndex: "1",
    };
  },
  methods: {},
};
</script>
<style scoped>

#Header {
  padding: 50px 50px;
  width: auto;
  margin: 1px 1em 10px 50px;

}

#word {

  height: 60px;
  line-height: 3.2em;
  letter-spacing: 8px;
  display: table-row;
}

h1 {
  /*text-align: center;*/
  color: #21b3b9;
  letter-spacing: 30px;
  font-size: 2.3em;
}

.el-menu-demo {
  width: 80%;
  margin: 0px auto;
  padding: 0px auto;
}

.top-left-edition{
  display: flex;
  float: left;
  margin-right: auto;
  margin-left: 20px;
}

i,
input,
label {
  vertical-align: middle;
}

i {
  border: 0;
  display: block;
  cursor: pointer;
}

/*.top-left-edition span {*/
/*  float: left;*/
/*  font-size: 16px;*/
/*  color: #999999;*/
/*  line-height: 24px;*/
/*  margin-right: 4px;*/
/*}*/
</style>


