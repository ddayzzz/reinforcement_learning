# coding=utf-8
"""
定义的浏览器接口，主要是操作浏览器获取 Dino 的canvas
"""

# 基本库的导入

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from utilities import show_img, grab_screen
import os
import threading


class Game:

    """
    游戏类，Python 和 浏览器的交互接口
    """
    def __init__(self, game_url="chrome://dino",
                 chrome_driver="./chromedriver",
                 init_script="document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'",
                 virtual_display=True):
        """
        构造函数
        :param game_url: Dino 的 URL
        :param chrome_driver: chrome driver 驱动程序
        """
        if not virtual_display:
            if os.name == 'nt':
                raise ValueError("Pyvirtualdisplay is not available in Windows")
            else:
                from pyvirtualdisplay import Display
                self._display = Display(visible=1)
                self._display.start()
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        self._driver = webdriver.Chrome(executable_path=chrome_driver, chrome_options=chrome_options)
        self._driver.set_window_position(x=-10, y=0)
        self._driver.get(game_url)
        self._driver.execute_script("Runner.config.ACCELERATION=0")
        self._driver.execute_script(init_script)

    def get_crashed(self):
        """
        Dino 是否已经死了（撞到了东西）
        :return: Bool
        """
        return self._driver.execute_script("return Runner.instance_.crashed")

    def get_playing(self):
        """
        Dino 是否处于玩的状态
        :return:
        """
        return self._driver.execute_script("return Runner.instance_.playing")

    def restart(self):
        """
        重新启动新的回合
        :return:
        """
        self._driver.execute_script("Runner.instance_.restart()")

    def press_up(self):
        """
        让 Dino 跳起（发送虚拟的 UP 键消息）
        :return:
        """
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def get_score(self):
        """
        获取得到的分数
        :return:
        """
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array)  # the javascript object is of type array with score in the formate[1,0,0] which is 100.
        return int(score)

    def pause(self):
        """
        暂停游戏
        :return:
        """
        return self._driver.execute_script("return Runner.instance_.stop()")

    def resume(self):
        """
        只要Dino没有死就继续 paused 的游戏
        :return:
        """
        return self._driver.execute_script("return Runner.instance_.play()")

    def stop(self):
        """
        暂停游戏
        :return:
        """
        return self._driver.execute_script("return Runner.instance_.stop()")

    def end(self):
        """
        结束一局游戏
        :return:
        """
        self._driver.close()
        # close virtual display
        if getattr(self, '_display'):
            self._display.stop()


class DinoAgent:

    """
    强化学习的 Agent
    """
    def __init__(self, game, start_immediately=True):
        """
        构造 Agent
        :param game: 利用 game 进行各种动作，最终反应到与 浏览器的接口上
        """
        self._game = game
        if start_immediately:
            self.jump()  # 首先起跳启动游戏

    def is_running(self):
        return self._game.get_playing()

    def is_crashed(self):
        return self._game.get_crashed()

    def jump(self):
        self._game.press_up()

    def duck(self):
        self._game.press_down()


class GameState:

    def __init__(self, agent, game, show=False):
        self._agent = agent
        self._game = game
        self.show = show
        if show:
            self._display = show_img()  # display the processed image on screen using openCV, implemented using python coroutine
            self._display.__next__()  # initiliaze the display coroutine

    def observe(self):
        image = grab_screen(self._game._driver,
                            getbase64Script="canvasRunner = document.getElementById('runner-canvas');return canvasRunner.toDataURL().substring(22)")
        if self.show:
            self._display.send(image)  # display the image on screen
        return image


def _start_a_game(game_state, game):

    def run():
        while True:
            if game.get_crashed():
                game.restart()
            game_state.observe()
    t = threading.Thread(target=run)
    t.start()


def testOutput():
    """
    用于测试，显示 Dino 能否在图片中正常显示
    :return:
    """
    # Dino 1
    game1 = Game(chrome_driver=r"/bin/chromedriver", virtual_display=True)
    dino1 = DinoAgent(game1)
    game_state1 = GameState(dino1, game1, show=True)
    _start_a_game(game_state1, game1)
    # _start_a_game(game_state2)
    input()

if __name__ == "__main__":
    testOutput()