<template>
  <div id="app" class="parent-contain"  >
    <Header></Header>
    <Content></Content>
    <Footer></Footer>
  </div>
</template>

<script>
import Header from "./components/Header";
import Footer from "./components/Footer";
import Content from "./components/Content";


export default {
  name: "aaa",
  components: {
    Header,
    Footer,
    Content,
  },

};
</script>
<style scoped>
</style>
<style scope="this api replaced by slot-scope in 2.5.0+">
</style>




