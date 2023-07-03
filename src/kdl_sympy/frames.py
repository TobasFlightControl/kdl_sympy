from __future__ import annotations  # 自クラスを返り値としてアノテートするために必要
from functools import singledispatchmethod
import numpy as np
import sympy
from sympy import Symbol, Matrix


class Vector:

    def __init__(self, x: Symbol, y: Symbol, z: Symbol) -> None:
        self.data = Matrix([x, y, z])

    @classmethod
    def Zero(cls) -> Vector:
        return cls(0, 0, 0)

    @classmethod
    def TransX(cls, x: Symbol) -> Vector:
        return cls(x, 0, 0)

    @classmethod
    def TransY(cls, y: Symbol) -> Vector:
        return cls(0, y, 0)

    @classmethod
    def TransZ(cls, z: Symbol) -> Vector:
        return cls(0, 0, z)

    def x(self) -> Symbol:
        return self.data[0]

    def y(self) -> Symbol:
        return self.data[1]

    def z(self) -> Symbol:
        return self.data[2]

    def norm(self, ord: int = 2) -> Symbol:
        """ ノルムを返す． """
        return self.data.norm(ord=ord)

    def normalize(self) -> Vector:
        """ 正規化する． """
        return self / self.norm(2)

    def cross_mat(self) -> Matrix:
        """ ベクトルの外積に相当する行列を返す． """
        x, y, z = self.data
        return Matrix([
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0],
        ])

    def __add__(self, rhs: Vector) -> Vector:
        x = self.x() + rhs.x()
        y = self.y() + rhs.y()
        z = self.z() + rhs.z()
        return Vector(x, y, z)

    def __sub__(self, rhs: Vector) -> Vector:
        x = self.x() - rhs.x()
        y = self.y() - rhs.y()
        z = self.z() - rhs.z()
        return Vector(x, y, z)

    def __truediv__(self, rhs: Symbol) -> Vector:
        x = self.x() / rhs
        y = self.y() / rhs
        z = self.z() / rhs
        return Vector(x, y, z)

    def __repr__(self) -> str:
        return f'x: {self.x()}, y: {self.y()}, z: {self.z()}'


class Rotation:

    def __init__(
        self,
        Xx: Symbol, Yx: Symbol, Zx: Symbol,
        Xy: Symbol, Yy: Symbol, Zy: Symbol,
        Xz: Symbol, Yz: Symbol, Zz: Symbol,
    ) -> None:
        self.data = Matrix([
            [Xx, Yx, Zx],
            [Xy, Yy, Zy],
            [Xz, Yz, Zz],
        ])

    @classmethod
    def Identity(cls) -> Rotation:
        return cls(1, 0, 0, 0, 1, 0, 0, 0, 1)

    @classmethod
    def RotX(cls, roll: Symbol) -> Rotation:
        cs = sympy.cos(roll)
        sn = sympy.sin(roll)
        return cls(1, 0, 0, 0, cs, -sn, 0, sn, cs)

    @classmethod
    def RotY(cls, pitch: Symbol) -> Rotation:
        cs = sympy.cos(pitch)
        sn = sympy.sin(pitch)
        return cls(cs, 0, sn, 0, 1, 0, -sn, 0, cs)

    @classmethod
    def RotZ(cls, yaw: Symbol) -> Rotation:
        cs = sympy.cos(yaw)
        sn = sympy.sin(yaw)
        return cls(cs, -sn, 0, sn, cs, 0, 0, 0, 1)

    @classmethod
    def RPY(cls, roll: Symbol, pitch: Symbol, yaw: Symbol) -> Rotation:
        return cls.RotZ(yaw) * cls.RotY(pitch) * cls.RotX(roll)

    @classmethod
    def Rodrigues(cls, axis: Vector, angle: Symbol) -> Rotation:
        """
        回転軸と回転角から回転行列を求める．
        ロボティクス(2.80)を使った方が速いが，ここではロドリゲスの公式通りに計算している．

        Parameters
        ----------
        axis: Vector
            回転軸
        angle: Symbol
            回転角[rad]

        Returns
        ----------
        Rotation
        """
        sn = sympy.sin(angle)
        cs = sympy.cos(angle)
        cross = axis.normalize().cross_mat()
        I = np.identity(3)  # sympy.Identity(3)だと続く行列和がMatAddのまま評価されなかった

        data = I + sn * cross + (1 - cs) * (cross @ cross)
        return cls(*data.flat())

    def inverse(self) -> Rotation:
        data = self.data.T
        return Rotation(*data.flat())

    @singledispatchmethod
    def __mul__(self, rhs: Rotation) -> Rotation:
        data = self.data @ rhs.data
        return Rotation(*data.flat())

    @__mul__.register
    def _(self, rhs: Vector) -> Vector:
        data = self.data @ rhs.data
        return Vector(*data)

    def __repr__(self) -> str:
        res = ''
        for r in range(3):
            for c in range(3):
                res += f'{self.data[r, c]}\t'
            res += '\n'
        return res


class Frame:
    """
    3次元空間中における並進と回転を表す．
    必ず並進->回転の順であることに注意．
    """

    def __init__(self, p: Vector, M: Rotation) -> None:
        self.p = p
        self.M = M

    @classmethod
    def Identity(cls) -> Frame:
        return cls(Vector.Zero(), Rotation.Identity())

    @classmethod
    def Trans(cls, p: Vector) -> None:
        return cls(p, Rotation.Identity())

    @classmethod
    def Rot(cls, M: Rotation) -> None:
        return cls(Vector.Zero(), M)

    @classmethod
    def TransX(cls, x: Symbol) -> Frame:
        return cls.Trans(Vector.TransX(x))

    @classmethod
    def TransY(cls, y: Symbol) -> Frame:
        return cls.Trans(Vector.TransY(y))

    @classmethod
    def TransZ(cls, z: Symbol) -> Frame:
        return cls.Trans(Vector.TransZ(z))

    @classmethod
    def RotX(cls, roll: Symbol) -> Frame:
        return cls.Rot(Rotation.RotX(roll))

    @classmethod
    def RotY(cls, pitch: Symbol) -> Frame:
        return cls.Rot(Rotation.RotY(pitch))

    @classmethod
    def RotZ(cls, yaw: Symbol) -> Frame:
        return cls.Rot(Rotation.RotZ(yaw))

    @classmethod
    def DH(cls, alpha: Symbol, a: Symbol, theta: Symbol, d: Symbol,) -> Frame:
        """
        DenavitHartenbergパラメータによる座標変換．

        Parameters
        ----------
        alpha: Symbol
            2軸の偏角[rad]
        a: Symbol
            2軸の法線距離
        theta: Symbol
            2法線の偏角[rad]
        d: Symbol
            2法線の距離

        Returns
        -------
        Frame
        """

        cs_alpha = sympy.cos(alpha)
        sn_alpha = sympy.sin(alpha)
        cs_theta = sympy.cos(theta)
        sn_theta = sympy.sin(theta)

        # ロボティクス(3.6)
        p = Matrix([a, -sn_alpha * d, cs_alpha * d])
        M = Matrix([
            [cs_theta, -sn_theta, 0],
            [sn_theta * cs_alpha, cs_theta * cs_alpha, -sn_alpha],
            [sn_theta * sn_alpha, cs_theta * sn_alpha, cs_alpha],
        ])

        return cls(p, M)

    def inverse(self) -> Frame:
        M_inv = self.M.inverse()
        return Frame(p=-M_inv @ self.p, M=M_inv)

    @singledispatchmethod
    def __mul__(self, rhs: Frame) -> Frame:
        p = self.M * rhs.p + self.p
        M = self.M * rhs.M
        return Frame(p=p, M=M)

    @__mul__.register
    def _(self, rhs: Vector) -> Vector:
        return self.M * rhs + self.p

    def __repr__(self) -> str:
        return f'p:\n{self.p.__repr__()}\n' + f'M:\n{self.M.__repr__()}\n'
